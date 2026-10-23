pub(super) fn pad(
    source: &[u8],
    width: u32,
    height: u32,
    target_width: u32,
    target_height: u32,
) -> Vec<u8> {
    let ratio = (target_width as f64 / width as f64).min(target_height as f64 / height as f64);
    let w = (width as f64 * ratio).round_ties_even().max(1.) as usize;
    let h = (height as f64 * ratio).round_ties_even().max(1.) as usize;
    let resized = super::super::rgb_resize::resize(source, width, height, w as u32, h as u32);
    let mut result = vec![0; target_width as usize * target_height as usize * 3];
    let left = (target_width as usize - w) / 2;
    let top = (target_height as usize - h) / 2;
    for y in 0..h {
        let offset = ((top + y) * target_width as usize + left) * 3;
        result[offset..offset + w * 3].copy_from_slice(&resized[y * w * 3..(y + 1) * w * 3]);
    }
    result
}
