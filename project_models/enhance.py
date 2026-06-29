def enhance(self, image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        return _bicubic_upscale(np.zeros((64,64),dtype=np.uint8), self.scale)

    if not self._try_load_model():
        return _bicubic_upscale(image, self.scale)

    try:
        patch_size = 256
        h, w = image.shape[:2]

        result = np.zeros(
            (h*self.scale, w*self.scale, 3),
            dtype=np.uint8
        )

        for y in range(0, h, patch_size):
            for x in range(0, w, patch_size):

                patch = image[
                    y:min(y+patch_size,h),
                    x:min(x+patch_size,w)
                ]

                tensor,_,_ = _prepare_tensor(
                    patch,
                    self.device
                )

                output = _pad_and_infer(
                    tensor,
                    self.model,
                    self.scale,
                    LIGHTWEIGHT_X2_CONFIG["window_size"]
                )

                sr = _tensor_to_bgr(output)

                sy = y*self.scale
                sx = x*self.scale

                result[
                    sy:sy+sr.shape[0],
                    sx:sx+sr.shape[1]
                ] = sr

                print(
                    f"Processed patch ({x},{y})"
                )

        return result

    except Exception as exc:
        print(
            f"SwinIR failed ({exc})"
        )
        return _bicubic_upscale(
            image,
            self.scale
        )