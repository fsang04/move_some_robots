"""SAM2 streaming segmenter for the live tracking port (REALTIME_SAM2_OVERVIEW.md §2.2).

Wraps the HuggingFace `Sam2VideoModel` streaming session -- the same code path
test_sam2_mask_dlo.py validated against the shipped ground-truth masks -- plus
the mask cleanup the wire tracker does NOT do itself on tracking frames
(morphological close + keep the largest component).

torch/transformers are imported lazily in __init__, so the replay path of the
live driver runs without the deep-learning stack installed.
"""
import numpy as np


def clean_mask(mask, close_ksize: int = 5, keep_largest: bool = True):
    """The cleanup a live segmenter must do itself: the tracker only filters
    components on the init frame (wire_tracker.py:274-276)."""
    import cv2
    out = (mask > 0).astype(np.uint8)
    if close_ksize > 0:
        k = np.ones((close_ksize, close_ksize), np.uint8)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
    if keep_largest:
        n, lab, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
        if n > 1:
            biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            out = (lab == biggest).astype(np.uint8)
    return out


class Sam2Segmenter:
    """Streaming SAM2: prompt once on the first frame, then one mask per frame.

    Usage:
        seg = Sam2Segmenter()
        mask0 = seg.segment(bgr0, prompt_points_xy=[(x1, y1), (x2, y2)])
        mask1 = seg.segment(bgr1)   # SAM2 memory propagates the object
    """

    def __init__(self, model: str = 'facebook/sam2.1-hiera-tiny', device: str = None,
                 mask_threshold: float = 0.0, close_ksize: int = 5,
                 keep_largest: bool = True):
        import torch
        from transformers import Sam2VideoModel, Sam2VideoProcessor

        # fp32 106.3 ms -> bf16+TF32 31.5 ms per frame (test_sam2_mask_dlo.py:41-44)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self._torch = torch
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.mask_threshold = mask_threshold
        self.close_ksize = close_ksize
        self.keep_largest = keep_largest
        self.model = Sam2VideoModel.from_pretrained(model, device_map=self.device).eval()
        self.processor = Sam2VideoProcessor.from_pretrained(model)
        self.session = self.processor.init_video_session(inference_device=self.device)
        self._frame_idx = 0

    def reset(self):
        """Forget the session (object memory + prompt). The next segment()
        call must give new prompt points. The model weights stay loaded."""
        self.session = self.processor.init_video_session(inference_device=self.device)
        self._frame_idx = 0

    def segment(self, bgr: np.ndarray, prompt_points_xy=None,
                neg_points_xy=None) -> np.ndarray:
        """(H,W,3) uint8 BGR -> cleaned (H,W) uint8 mask, values 0 and 1.

        prompt_points_xy: iterable of (x, y) positive points on the object.
        Required on the first call, forbidden afterwards.
        neg_points_xy: optional (x, y) NEGATIVE points ("not the object") --
        e.g. on the gripper so the streamed masks stop where the cable is 
        grasped instead of extending into the EE wrist. First call only.
        """
        import cv2
        torch = self._torch

        if (prompt_points_xy is not None) != (self._frame_idx == 0):
            raise ValueError('give prompt_points_xy on the first frame, and only there')
        if neg_points_xy is not None and prompt_points_xy is None:
            raise ValueError('neg_points_xy only works alongside prompt_points_xy')

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)   # SAM2 wants RGB; sources give BGR
        inputs = self.processor(images=rgb, device=self.device,
                                return_tensors='pt').to(self.model.device)

        if prompt_points_xy is not None:
            pts = np.asarray(prompt_points_xy, dtype=np.float64)
            labels = np.ones(len(pts), dtype=np.int32)
            if neg_points_xy is not None and len(neg_points_xy):
                negs = np.asarray(neg_points_xy, dtype=np.float64)
                pts = np.vstack([pts, negs])
                labels = np.concatenate([labels, np.zeros(len(negs), dtype=np.int32)])
            self.processor.add_inputs_to_inference_session(
                inference_session=self.session,
                frame_idx=0,
                obj_ids=1,
                input_points=[[pts.tolist()]],                       # [batch][obj][pt][xy]
                input_labels=[[labels.tolist()]],                    # 1 = object, 0 = not
                original_size=inputs.original_sizes[0],              # required when streaming
            )

        with torch.inference_mode():
            if self.device == 'cuda':
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    out = self.model(inference_session=self.session,
                                     frame=inputs.pixel_values[0])
            else:
                out = self.model(inference_session=self.session,
                                 frame=inputs.pixel_values[0])
        logits = self.processor.post_process_masks(
            [out.pred_masks], original_sizes=inputs.original_sizes, binarize=False)[0]
        raw = (logits[0, 0].float().cpu().numpy() > self.mask_threshold).astype(np.uint8)

        H, W = bgr.shape[:2]
        if raw.shape != (H, W):
            raw = cv2.resize(raw, (W, H), interpolation=cv2.INTER_NEAREST)
        self._frame_idx += 1
        return clean_mask(raw, self.close_ksize, self.keep_largest)
