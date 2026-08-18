use candle_core::{
    cuda::{
        cudarc::driver::{DevicePtr, LaunchAsync, LaunchConfig},
        CudaError, WrapErr, S,
    },
    shape::Dim,
    CudaStorage, DType, Error, Result, Storage, Tensor,
};

use crate::kernels;

// A random number.
pub const COPY2D_FINGERPRINT: u64 = 7472013941709904021;

trait CudaStorageExtension {
    #[allow(clippy::too_many_arguments)]
    fn copy2d_fingerprinted(
        &self,
        src: &CudaStorage,
        dst: &CudaStorage,
        d1: usize,
        d2: usize,
        src_s: usize,
        dst_s: usize,
        src_o: usize,
        dst_o: usize,
    ) -> Result<()>;

    #[allow(clippy::too_many_arguments)]
    fn copy2d_fingerprinted_dynoffset(
        &self,
        src: &CudaStorage,
        dst: &CudaStorage,
        d1: usize,
        d2: usize,
        src_s: usize,
        dst_s: usize,
        src_o: usize,
        dst_o_base: usize,
        dst_o_stride: usize,
        dst_o_ptr: &CudaStorage,
    ) -> Result<()>;
}

/// Provides extensions to Tensor.
pub trait CudaTensorExtension {
    /// `slice_set`.
    ///
    /// The `copy2d_*` function takes a u64 fingerprint as the first argument, specifically [`COPY2D_FINGERPRINT`].
    fn slice_set_fingerprinted<D: Dim>(&self, src: &Self, dim: D, offset: usize) -> Result<()>;

    /// Like [`slice_set_fingerprinted`](Self::slice_set_fingerprinted), except the offset on
    /// `dim` is read from `position` (a `U32` tensor holding exactly one element) at kernel
    /// launch time instead of being passed as a plain `usize`.
    ///
    /// A plain offset becomes a fixed kernel-launch argument once captured into a CUDA graph,
    /// so every replay would write to the same slot the capture did. Reading the offset from
    /// `position` at launch time instead means only the *pointer* is captured; updating
    /// `position`'s contents between replays (e.g. via [`copy_inplace`](crate::copy_inplace))
    /// changes where the next replay writes without re-capturing anything. See the
    /// `decode_replay` example.
    fn slice_set_fingerprinted_at<D: Dim>(
        &self,
        src: &Self,
        dim: D,
        offset: usize,
        position: &Self,
    ) -> Result<()>;
}

impl CudaTensorExtension for Tensor {
    /// Set the values on `self` using values from `src`. The copy starts at the specified
    /// `offset` for the target dimension `dim` on `self`.
    /// `self` and `src` must have the same shape except on dimension `dim` where the `self` size
    /// has to be greater than or equal to `offset` plus the `src` size.
    ///
    /// Note that this modifies `self` in place and as such is not compatibel with
    /// back-propagation.  
    fn slice_set_fingerprinted<D: Dim>(&self, src: &Self, dim: D, offset: usize) -> Result<()> {
        let dim = dim.to_index(self.shape(), "slice-set")?;
        if !self.is_contiguous() || !src.is_contiguous() {
            Err(Error::RequiresContiguous { op: "slice-set" }.bt())?
        }
        if self.dtype() != src.dtype() {
            Err(Error::DTypeMismatchBinaryOp {
                lhs: self.dtype(),
                rhs: src.dtype(),
                op: "slice-set",
            }
            .bt())?
        }
        if self.device().location() != src.device().location() {
            Err(Error::DeviceMismatchBinaryOp {
                lhs: self.device().location(),
                rhs: src.device().location(),
                op: "slice-set",
            }
            .bt())?
        }
        if self.rank() != src.rank() {
            Err(Error::UnexpectedNumberOfDims {
                expected: self.rank(),
                got: src.rank(),
                shape: self.shape().clone(),
            }
            .bt())?
        }
        for (dim_idx, (v1, v2)) in self.dims().iter().zip(src.dims().iter()).enumerate() {
            if dim_idx == dim && *v2 + offset > *v1 {
                candle_core::bail!("shape mismatch on target dim, dst: {v1}, src: {v2} + {offset}")
            }
            if dim_idx != dim && v1 != v2 {
                candle_core::bail!("shape mismatch on dim {dim_idx}, {v1} <> {v2}")
            }
        }
        let block_size: usize = src.dims().iter().skip(1 + dim).product();
        let d1: usize = src.dims().iter().take(dim).product();
        let d2 = block_size * src.dims()[dim];
        let dst_o = self.layout().start_offset() + offset * block_size;
        let src_o = src.layout().start_offset();
        match (&*self.storage_and_layout().0, &*src.storage_and_layout().0) {
            (Storage::Cuda(src), Storage::Cuda(dst)) => self.copy2d_fingerprinted(
                dst,
                src,
                d1,
                d2,
                d2,
                block_size * self.dims()[dim],
                src_o,
                dst_o,
            ),
            (lhs, rhs) => Err(Error::DeviceMismatchBinaryOp {
                lhs: lhs.device().location(),
                rhs: rhs.device().location(),
                op: "copy2d CUDA",
            }
            .bt()),
        }
    }

    fn slice_set_fingerprinted_at<D: Dim>(
        &self,
        src: &Self,
        dim: D,
        offset: usize,
        position: &Self,
    ) -> Result<()> {
        let dim = dim.to_index(self.shape(), "slice-set-at")?;
        if !self.is_contiguous() || !src.is_contiguous() {
            Err(Error::RequiresContiguous { op: "slice-set-at" }.bt())?
        }
        if self.dtype() != src.dtype() {
            Err(Error::DTypeMismatchBinaryOp {
                lhs: self.dtype(),
                rhs: src.dtype(),
                op: "slice-set-at",
            }
            .bt())?
        }
        if self.device().location() != src.device().location()
            || self.device().location() != position.device().location()
        {
            Err(Error::DeviceMismatchBinaryOp {
                lhs: self.device().location(),
                rhs: src.device().location(),
                op: "slice-set-at",
            }
            .bt())?
        }
        if self.rank() != src.rank() {
            Err(Error::UnexpectedNumberOfDims {
                expected: self.rank(),
                got: src.rank(),
                shape: self.shape().clone(),
            }
            .bt())?
        }
        if position.dtype() != DType::U32 {
            candle_core::bail!(
                "slice-set-at: position must be U32, got {:?}",
                position.dtype()
            )
        }
        if position.elem_count() != 1 {
            candle_core::bail!(
                "slice-set-at: position must hold exactly one element, got shape {:?}",
                position.shape()
            )
        }
        if !position.is_contiguous() {
            Err(Error::RequiresContiguous { op: "slice-set-at" }.bt())?
        }
        for (dim_idx, (v1, v2)) in self.dims().iter().zip(src.dims().iter()).enumerate() {
            // `position` is only known at kernel-launch time, so this can't also bounds-check
            // `*position + src` against `self`'s size on `dim` the way `slice_set_fingerprinted`
            // does for its host-known `offset` -- that check has to live at the call site
            // (e.g. `Cache::append_at`'s `max_seq_len` bound), or move into the kernel itself.
            if dim_idx != dim && v1 != v2 {
                candle_core::bail!("shape mismatch on dim {dim_idx}, {v1} <> {v2}")
            }
        }
        let block_size: usize = src.dims().iter().skip(1 + dim).product();
        let d1: usize = src.dims().iter().take(dim).product();
        let d2 = block_size * src.dims()[dim];
        let dst_o_base = self.layout().start_offset() + offset * block_size;
        let src_o = src.layout().start_offset();
        match (
            &*self.storage_and_layout().0,
            &*src.storage_and_layout().0,
            &*position.storage_and_layout().0,
        ) {
            (Storage::Cuda(dst), Storage::Cuda(src), Storage::Cuda(position)) => self
                .copy2d_fingerprinted_dynoffset(
                    src,
                    dst,
                    d1,
                    d2,
                    d2,
                    block_size * self.dims()[dim],
                    src_o,
                    dst_o_base,
                    block_size,
                    position,
                ),
            _ => Err(Error::DeviceMismatchBinaryOp {
                lhs: self.device().location(),
                rhs: src.device().location(),
                op: "copy2d CUDA",
            }
            .bt()),
        }
    }
}

impl CudaStorageExtension for Tensor {
    fn copy2d_fingerprinted(
        &self,
        src: &CudaStorage,
        dst: &CudaStorage,
        d1: usize,
        d2: usize,
        src_s: usize,
        dst_s: usize,
        src_o: usize,
        dst_o: usize,
    ) -> Result<()> {
        let dev = &src.device;
        let d1 = d1 as u32;
        let d2 = d2 as u32;
        // Nothing to copy so we exit early to avoid launching a kernel and some potential invalid
        // argument with a null pointer.
        if d1 == 0 || d2 == 0 {
            return Ok(());
        }
        let dst_s = dst_s as u32;
        let src_s = src_s as u32;
        let (src, dst, kname) = match (&src.slice, &dst.slice) {
            (S::U8(s), S::U8(d)) => (*s.device_ptr(), *d.device_ptr(), "copy2d_u8"),
            (S::U32(s), S::U32(d)) => (*s.device_ptr(), *d.device_ptr(), "copy2d_u32"),
            (S::I64(s), S::I64(d)) => (*s.device_ptr(), *d.device_ptr(), "copy2d_i64"),
            (S::BF16(s), S::BF16(d)) => (*s.device_ptr(), *d.device_ptr(), "copy2d_bf16"),
            (S::F16(s), S::F16(d)) => (*s.device_ptr(), *d.device_ptr(), "copy2d_f16"),
            (S::F32(s), S::F32(d)) => (*s.device_ptr(), *d.device_ptr(), "copy2d_f32"),
            (S::F64(s), S::F64(d)) => (*s.device_ptr(), *d.device_ptr(), "copy2d_f64"),
            _ => Err(CudaError::InternalError("dtype mismatch in copy2d"))?,
        };
        let func = dev.get_or_load_func(kname, kernels::UPDATE_KV)?;
        let cfg = LaunchConfig::for_num_elems(d1 * d2);
        let params = (
            COPY2D_FINGERPRINT,
            src,
            dst,
            d1,
            d2,
            src_o,
            dst_o,
            src_s,
            dst_s,
        );
        // SAFETY: ffi.
        unsafe { func.launch(cfg, params) }.w()?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn copy2d_fingerprinted_dynoffset(
        &self,
        src: &CudaStorage,
        dst: &CudaStorage,
        d1: usize,
        d2: usize,
        src_s: usize,
        dst_s: usize,
        src_o: usize,
        dst_o_base: usize,
        dst_o_stride: usize,
        dst_o_ptr: &CudaStorage,
    ) -> Result<()> {
        let dev = &src.device;
        let d1 = d1 as u32;
        let d2 = d2 as u32;
        // Nothing to copy so we exit early to avoid launching a kernel and some potential invalid
        // argument with a null pointer.
        if d1 == 0 || d2 == 0 {
            return Ok(());
        }
        let dst_s = dst_s as u32;
        let src_s = src_s as u32;
        let dst_o_base = dst_o_base as u32;
        let dst_o_stride = dst_o_stride as u32;
        // `slice_set_fingerprinted_at` already checked `position.dtype() == DType::U32` before
        // calling here; this match is exhaustive-by-construction, not a runtime dtype guard.
        let position_ptr = match &dst_o_ptr.slice {
            S::U32(p) => *p.device_ptr(),
            _ => Err(CudaError::InternalError(
                "slice-set-at: position storage must be U32",
            ))?,
        };
        let (src, dst, kname) = match (&src.slice, &dst.slice) {
            (S::U8(s), S::U8(d)) => (*s.device_ptr(), *d.device_ptr(), "copy2d_dynoffset_u8"),
            (S::U32(s), S::U32(d)) => (*s.device_ptr(), *d.device_ptr(), "copy2d_dynoffset_u32"),
            (S::I64(s), S::I64(d)) => (*s.device_ptr(), *d.device_ptr(), "copy2d_dynoffset_i64"),
            (S::BF16(s), S::BF16(d)) => (*s.device_ptr(), *d.device_ptr(), "copy2d_dynoffset_bf16"),
            (S::F16(s), S::F16(d)) => (*s.device_ptr(), *d.device_ptr(), "copy2d_dynoffset_f16"),
            (S::F32(s), S::F32(d)) => (*s.device_ptr(), *d.device_ptr(), "copy2d_dynoffset_f32"),
            (S::F64(s), S::F64(d)) => (*s.device_ptr(), *d.device_ptr(), "copy2d_dynoffset_f64"),
            _ => Err(CudaError::InternalError(
                "dtype mismatch in copy2d_dynoffset",
            ))?,
        };
        let func = dev.get_or_load_func(kname, kernels::UPDATE_KV)?;
        let cfg = LaunchConfig::for_num_elems(d1 * d2);
        let params = (
            COPY2D_FINGERPRINT,
            src,
            dst,
            d1,
            d2,
            src_o,
            dst_o_base,
            dst_o_stride,
            src_s,
            dst_s,
            position_ptr,
        );
        // SAFETY: ffi.
        unsafe { func.launch(cfg, params) }.w()?;
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use std::collections::HashMap;

    use candle_core::{DType, Device, Result, Tensor};

    use crate::{
        extension::CudaTensorExtension,
        graph::{copy_inplace, Graph, GraphInput},
    };

    #[test]
    fn slice_set() -> Result<()> {
        let device = Device::new_cuda_with_stream(0)?;

        let (b, h, max_t, d) = (2, 4, 7, 3);
        let cache = Tensor::zeros((b, h, max_t, d), DType::F32, &device)?;
        let tensor = Tensor::randn(0f32, 1f32, (b, h, 4, d), &device)?;
        cache.slice_set_fingerprinted(&tensor, 2, 0)?;
        let cache_t = cache.narrow(2, 0, 4)?;
        let diff = (cache_t - &tensor)?.abs()?.sum_all()?.to_vec0::<f32>()?;
        assert_eq!(diff, 0.);
        cache.slice_set_fingerprinted(&tensor, 2, 1)?;
        let cache_t = cache.narrow(2, 1, 4)?;
        let diff = (cache_t - &tensor)?.abs()?.sum_all()?.to_vec0::<f32>()?;
        assert_eq!(diff, 0.);
        let ones = Tensor::ones((b, h, 1, d), DType::F32, &device)?;
        cache.slice_set_fingerprinted(&ones, 2, 6)?;
        let diff = cache.narrow(2, 5, 1)?.abs()?.sum_all()?.to_vec0::<f32>()?;
        assert_eq!(diff, 0.);
        let diff = (cache.narrow(2, 6, 1)? - 1.)?
            .abs()?
            .sum_all()?
            .to_vec0::<f32>()?;
        assert_eq!(diff, 0.);
        Ok(())
    }

    #[test]
    fn slice_set_at_matches_slice_set() -> Result<()> {
        let device = Device::new_cuda_with_stream(0)?;

        let (b, h, max_t, d) = (2, 4, 7, 3);
        let cache = Tensor::zeros((b, h, max_t, d), DType::F32, &device)?;
        let tensor = Tensor::randn(0f32, 1f32, (b, h, 1, d), &device)?;
        let position = Tensor::new(&[3u32], &device)?;

        cache.slice_set_fingerprinted_at(&tensor, 2, 0, &position)?;
        let cache_t = cache.narrow(2, 3, 1)?;
        let diff = (cache_t - &tensor)?.abs()?.sum_all()?.to_vec0::<f32>()?;
        assert_eq!(diff, 0.);

        // Every other slot must be untouched.
        for other in [0usize, 1, 2, 4, 5, 6] {
            let diff = cache
                .narrow(2, other, 1)?
                .abs()?
                .sum_all()?
                .to_vec0::<f32>()?;
            assert_eq!(diff, 0., "slot {other} should still be zero");
        }
        Ok(())
    }

    /// The bug this module exists to fix: without a device-resident position, a
    /// `slice_set_fingerprinted` call captured into a graph writes to the same
    /// slot on every replay, because the offset is baked into the graph as a
    /// plain kernel-launch argument at capture time. This test captures a graph
    /// containing a single `slice_set_fingerprinted_at` call once, replays it
    /// three times with `position` updated to 0, 1, 2 between replays (never
    /// re-capturing), and asserts each replay landed in its own slot -- i.e.
    /// that the write position tracks graph *replay* time, not capture time.
    #[test]
    fn slice_set_at_survives_graph_replay() -> anyhow::Result<()> {
        struct Inputs {
            x: Tensor,
            position: Tensor,
        }

        impl GraphInput for Inputs {
            fn to_inputs(&self) -> HashMap<String, Tensor> {
                [
                    ("x".to_string(), self.x.clone()),
                    ("position".to_string(), self.position.clone()),
                ]
                .into()
            }
            fn load_inputs_inplace(&self, input: Self, device: &Device) -> anyhow::Result<()> {
                unsafe {
                    copy_inplace(&input.x, &self.x, device)?;
                    copy_inplace(&input.position, &self.position, device)?;
                }
                Ok(())
            }
        }

        let device = Device::new_cuda_with_stream(0)?;
        let (b, h, max_t, d) = (1, 1, 4, 2);

        let cache = Tensor::zeros((b, h, max_t, d), DType::F32, &device)?;
        let x = Tensor::zeros((b, h, 1, d), DType::F32, &device)?;
        let position = Tensor::new(&[0u32], &device)?;

        let graph = Graph::new(
            |input: &Inputs| {
                cache.slice_set_fingerprinted_at(&input.x, 2, 0, &input.position)?;
                Ok(())
            },
            &device,
            Inputs { x, position },
        )?;

        for step in 0u32..3 {
            // `Tensor::full` broadcasts a 1-element storage, so it has to be materialized
            // before `copy_inplace` (which memcpy's the raw storage) can use it.
            let x = Tensor::full(step as f32 + 1., (b, h, 1, d), &device)?.contiguous()?;
            let position = Tensor::new(&[step], &device)?;
            graph.replay(Inputs { x, position })?;

            let written = cache
                .narrow(2, step as usize, 1)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            for &v in &written {
                assert_eq!(v, step as f32 + 1., "slot {step} after replay {step}");
            }
        }

        // The final replay must not have touched earlier slots -- if the offset
        // were still pinned at capture time (0), every replay would have
        // overwritten slot 0 instead of advancing.
        for slot in 0usize..2 {
            let expected = slot as f32 + 1.;
            let got = cache.narrow(2, slot, 1)?.flatten_all()?.to_vec1::<f32>()?;
            for &v in &got {
                assert_eq!(
                    v, expected,
                    "slot {slot} must retain its own replay's value"
                );
            }
        }

        Ok(())
    }
}
