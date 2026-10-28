//! The generic paint canvas: marks over a reference (a translucent layer sent as a
//! marked copy) or a region map (flat colors, black unspecified). Pure pixel work,
//! independent of egui, so its edits are testable.

/// Long side of the canvas; larger images are painted downscaled.
pub const MAX_SIDE: usize = 2048;
/// Undo steps kept.
const HISTORY: usize = 10;
/// Base-color distance a marks fill spreads over.
const FILL_TOLERANCE: u8 = 24;
const BLACK: [u8; 4] = [0, 0, 0, 255];
const CLEAR: [u8; 4] = [0, 0, 0, 0];
/// Distinct region colors for new regions, none black.
pub const REGION_PALETTE: [[u8; 3]; 12] = [
    [230, 25, 75],
    [60, 180, 75],
    [255, 225, 25],
    [0, 130, 200],
    [245, 130, 48],
    [145, 30, 180],
    [70, 240, 240],
    [240, 50, 230],
    [210, 245, 60],
    [250, 190, 212],
    [0, 128, 128],
    [170, 110, 40],
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PaintKind {
    Marks,
    Regions,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tool {
    Brush,
    Fill,
    Erase,
    Pick,
}
/// RGBA pixels, unpremultiplied.
#[derive(Clone, Debug, PartialEq)]
pub struct Pixels {
    pub width: usize,
    pub height: usize,
    pub rgba: Vec<[u8; 4]>,
}
impl Pixels {
    pub fn new(width: usize, height: usize, fill: [u8; 4]) -> Self {
        Self {
            width,
            height,
            rgba: vec![fill; width * height],
        }
    }
    pub fn from_rgba(width: usize, height: usize, bytes: &[u8]) -> Self {
        Self {
            width,
            height,
            rgba: bytes
                .chunks_exact(4)
                .map(|p| [p[0], p[1], p[2], p[3]])
                .collect(),
        }
    }
    pub fn bytes(&self) -> Vec<u8> {
        self.rgba.iter().flatten().copied().collect()
    }
    /// Scaled to fit `MAX_SIDE` (Lanczos), unchanged when it fits.
    pub fn fitted(self) -> Self {
        let long = self.width.max(self.height);
        if long <= MAX_SIDE {
            return self;
        }
        let scale = MAX_SIDE as f64 / long as f64;
        let (w, h) = (
            ((self.width as f64 * scale).round() as u32).max(1),
            ((self.height as f64 * scale).round() as u32).max(1),
        );
        let image = image::RgbaImage::from_raw(self.width as u32, self.height as u32, self.bytes())
            .expect("matching pixel count");
        let image = image::imageops::resize(&image, w, h, image::imageops::FilterType::Lanczos3);
        Self::from_rgba(w as usize, h as usize, image.as_raw())
    }
    /// PNG bytes.
    pub fn png(&self) -> anyhow::Result<Vec<u8>> {
        let mut out = std::io::Cursor::new(Vec::new());
        image::RgbaImage::from_raw(self.width as u32, self.height as u32, self.bytes())
            .ok_or_else(|| anyhow::anyhow!("Invalid canvas size."))?
            .write_to(&mut out, image::ImageFormat::Png)?;
        Ok(out.into_inner())
    }
}
pub struct Canvas {
    pub kind: PaintKind,
    /// The picture under the paint: the reference for marks, the source photo
    /// (when known) for regions.
    pub base: Option<Pixels>,
    /// Marks with their opacity, or the region map itself.
    pub layer: Pixels,
    undo: Vec<Vec<[u8; 4]>>,
    redo: Vec<Vec<[u8; 4]>>,
}
impl Canvas {
    /// Marks over `base`, continuing `layer` when the reference was marked before.
    pub fn marks(base: Pixels, layer: Option<Pixels>) -> Self {
        let layer = layer
            .filter(|l| l.width == base.width && l.height == base.height)
            .unwrap_or_else(|| Pixels::new(base.width, base.height, CLEAR));
        Self {
            kind: PaintKind::Marks,
            base: Some(base),
            layer,
            undo: Vec::new(),
            redo: Vec::new(),
        }
    }
    /// A region map, opaque: transparent pixels count as unspecified.
    pub fn regions(mut map: Pixels, base: Option<Pixels>) -> Self {
        for p in &mut map.rgba {
            *p = if p[3] < 128 {
                BLACK
            } else {
                [p[0], p[1], p[2], 255]
            };
        }
        Self {
            kind: PaintKind::Regions,
            base: base.filter(|b| b.width == map.width && b.height == map.height),
            layer: map,
            undo: Vec::new(),
            redo: Vec::new(),
        }
    }
    pub fn size(&self) -> [usize; 2] {
        [self.layer.width, self.layer.height]
    }
    /// What erasing paints: nothing for marks, unspecified for regions.
    pub fn erase_color(&self) -> [u8; 4] {
        match self.kind {
            PaintKind::Marks => CLEAR,
            PaintKind::Regions => BLACK,
        }
    }
    /// Starts an edit that one undo reverts.
    pub fn begin(&mut self) {
        self.undo.push(self.layer.rgba.clone());
        if self.undo.len() > HISTORY {
            self.undo.remove(0);
        }
        self.redo.clear();
    }
    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }
    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }
    pub fn undo(&mut self) -> bool {
        let Some(previous) = self.undo.pop() else {
            return false;
        };
        self.redo
            .push(std::mem::replace(&mut self.layer.rgba, previous));
        true
    }
    pub fn redo(&mut self) -> bool {
        let Some(next) = self.redo.pop() else {
            return false;
        };
        self.undo
            .push(std::mem::replace(&mut self.layer.rgba, next));
        true
    }
    /// A round brush of `radius` pixels from `from` to `to` (canvas pixels).
    pub fn line(&mut self, from: [f32; 2], to: [f32; 2], radius: f32, color: [u8; 4]) {
        let radius = radius.max(0.5);
        let length = ((to[0] - from[0]).powi(2) + (to[1] - from[1]).powi(2)).sqrt();
        let steps = (length / (radius / 3.).max(0.5)).ceil().max(1.) as usize;
        for i in 0..=steps {
            let t = i as f32 / steps as f32;
            self.dot(
                [
                    from[0] + (to[0] - from[0]) * t,
                    from[1] + (to[1] - from[1]) * t,
                ],
                radius,
                color,
            );
        }
    }
    fn dot(&mut self, center: [f32; 2], radius: f32, color: [u8; 4]) {
        let [w, h] = self.size();
        let y0 = (center[1] - radius).floor().max(0.) as usize;
        let y1 = ((center[1] + radius).ceil() as usize).min(h);
        let x0 = (center[0] - radius).floor().max(0.) as usize;
        let x1 = ((center[0] + radius).ceil() as usize).min(w);
        for y in y0..y1 {
            for x in x0..x1 {
                let (dx, dy) = (x as f32 + 0.5 - center[0], y as f32 + 0.5 - center[1]);
                if dx * dx + dy * dy <= radius * radius {
                    self.layer.rgba[y * w + x] = color;
                }
            }
        }
    }
    /// Fills from `at`: the connected region of one color in a region map, or the
    /// connected area of similar color in the reference for marks.
    pub fn fill(&mut self, at: [usize; 2], color: [u8; 4]) {
        let [w, h] = self.size();
        if at[0] >= w || at[1] >= h {
            return;
        }
        let seed = at[1] * w + at[0];
        let source = match (self.kind, &self.base) {
            (PaintKind::Marks, Some(base)) => &base.rgba,
            _ => &self.layer.rgba,
        };
        let target = source[seed];
        let tolerance = match self.kind {
            PaintKind::Marks => FILL_TOLERANCE,
            PaintKind::Regions => 0,
        };
        let similar = |p: [u8; 4]| (0..3).all(|c| p[c].abs_diff(target[c]) <= tolerance);
        let mut inside = vec![false; w * h];
        let mut stack = vec![seed];
        inside[seed] = true;
        while let Some(i) = stack.pop() {
            let (x, y) = (i % w, i / w);
            let neighbors = [
                (x > 0).then(|| i - 1),
                (x + 1 < w).then(|| i + 1),
                (y > 0).then(|| i - w),
                (y + 1 < h).then(|| i + w),
            ];
            for n in neighbors.into_iter().flatten() {
                if !inside[n] && similar(source[n]) {
                    inside[n] = true;
                    stack.push(n);
                }
            }
        }
        for (pixel, inside) in self.layer.rgba.iter_mut().zip(inside) {
            if inside {
                *pixel = color;
            }
        }
    }
    /// The region color under `at`, `None` where unspecified or outside.
    pub fn pick(&self, at: [usize; 2]) -> Option<[u8; 3]> {
        let [w, h] = self.size();
        if at[0] >= w || at[1] >= h {
            return None;
        }
        let p = self.layer.rgba[at[1] * w + at[0]];
        (p != BLACK && p[3] > 0).then_some([p[0], p[1], p[2]])
    }
    /// The region map's colors by area, largest first, without unspecified black.
    pub fn region_list(&self) -> Vec<([u8; 3], f32)> {
        let mut counts = std::collections::HashMap::<[u8; 3], usize>::new();
        for p in &self.layer.rgba {
            if *p != BLACK {
                *counts.entry([p[0], p[1], p[2]]).or_default() += 1;
            }
        }
        let total = self.layer.rgba.len().max(1) as f32;
        let mut list: Vec<_> = counts
            .into_iter()
            .map(|(color, n)| (color, n as f32 / total))
            .collect();
        list.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        list
    }
    /// Repaints every pixel of region `from` in `to` (merging or removing it).
    pub fn recolor(&mut self, from: [u8; 3], to: [u8; 4]) {
        for p in &mut self.layer.rgba {
            if [p[0], p[1], p[2]] == from && p[3] == 255 {
                *p = to;
            }
        }
    }
    /// The first palette color no region uses yet.
    pub fn new_region_color(&self) -> [u8; 3] {
        let used: Vec<_> = self.region_list().into_iter().map(|(c, _)| c).collect();
        REGION_PALETTE
            .into_iter()
            .find(|c| !used.contains(c))
            .unwrap_or(REGION_PALETTE[used.len() % REGION_PALETTE.len()])
    }
    /// Removes all marks, or all regions.
    pub fn clear(&mut self) {
        let color = self.erase_color();
        self.layer.rgba.fill(color);
    }
    pub fn has_marks(&self) -> bool {
        self.layer.rgba.iter().any(|p| p[3] > 0)
    }
    /// The image to send: the reference with its marks, or the region map.
    pub fn composite(&self) -> Pixels {
        match (self.kind, &self.base) {
            (PaintKind::Marks, Some(base)) => {
                let rgba = base
                    .rgba
                    .iter()
                    .zip(&self.layer.rgba)
                    .map(|(b, m)| {
                        let a = m[3] as u32;
                        let mix = |c: usize| {
                            ((m[c] as u32 * a + b[c] as u32 * (255 - a) + 127) / 255) as u8
                        };
                        [mix(0), mix(1), mix(2), 255]
                    })
                    .collect();
                Pixels {
                    width: base.width,
                    height: base.height,
                    rgba,
                }
            }
            _ => self.layer.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marks_paint_fill_by_reference_color_and_blend_into_a_copy() {
        let mut base = Pixels::new(8, 4, [200, 200, 200, 255]);
        for x in 4..8 {
            for y in 0..4 {
                base.rgba[y * 8 + x] = [20, 40, 60, 255];
            }
        }
        let mut canvas = Canvas::marks(base.clone(), None);
        assert!(!canvas.has_marks());
        canvas.begin();
        canvas.fill([6, 1], [255, 0, 0, 255]);
        assert_eq!(canvas.layer.rgba.iter().filter(|p| p[3] == 255).count(), 16);
        let copy = canvas.composite();
        assert_eq!(copy.rgba[0], [200, 200, 200, 255]);
        assert_eq!(copy.rgba[7], [255, 0, 0, 255]);
        // Half-opaque marks blend; erasing clears them.
        canvas.begin();
        canvas.line([0.5, 0.5], [0.5, 0.5], 0.6, [0, 0, 255, 128]);
        assert_eq!(canvas.composite().rgba[0], [100, 100, 228, 255]);
        canvas.line([0.5, 0.5], [0.5, 0.5], 0.6, canvas.erase_color());
        assert_eq!(canvas.layer.rgba[0], CLEAR);
        // The original stays untouched; undo and redo restore whole edits.
        assert_eq!(canvas.base.as_ref(), Some(&base));
        assert!(canvas.undo() && canvas.undo());
        assert!(!canvas.has_marks());
        assert!(canvas.redo());
        assert!(canvas.has_marks());
        canvas.clear();
        assert!(!canvas.has_marks());
    }

    #[test]
    fn region_maps_stay_opaque_list_regions_and_merge_remove_and_add_them() {
        let mut map = Pixels::new(4, 2, [230, 25, 75, 255]);
        map.rgba[3] = [0, 0, 0, 0];
        map.rgba[4] = [60, 180, 75, 255];
        let mut canvas = Canvas::regions(map, None);
        assert_eq!(canvas.layer.rgba[3], BLACK);
        assert_eq!(
            canvas.region_list(),
            vec![([230, 25, 75], 6. / 8.), ([60, 180, 75], 1. / 8.)]
        );
        assert_eq!(canvas.pick([0, 1]), Some([60, 180, 75]));
        assert_eq!(canvas.pick([3, 0]), None);
        assert_eq!(canvas.new_region_color(), [255, 225, 25]);
        // Fill takes the connected region of one exact color only.
        canvas.begin();
        canvas.fill([0, 1], [255, 225, 25, 255]);
        assert_eq!(canvas.layer.rgba[4], [255, 225, 25, 255]);
        assert_eq!(canvas.layer.rgba[5], [230, 25, 75, 255]);
        canvas.recolor([255, 225, 25], [230, 25, 75, 255]);
        assert_eq!(canvas.region_list().len(), 1);
        canvas.recolor([230, 25, 75], canvas.erase_color());
        assert!(canvas.region_list().is_empty());
        canvas.line([0., 0.], [4., 2.], 0.8, [0, 130, 200, 255]);
        assert!(canvas.layer.rgba[0] == [0, 130, 200, 255]);
        assert_eq!(canvas.composite(), canvas.layer);
    }

    #[test]
    fn large_images_paint_downscaled_and_export_png() {
        let big = Pixels::new(4096, 1024, [10, 20, 30, 255]).fitted();
        assert_eq!((big.width, big.height), (MAX_SIDE, 512));
        let png = Pixels::new(3, 2, [1, 2, 3, 255]).png().unwrap();
        let back = image::load_from_memory(&png).unwrap().to_rgba8();
        assert_eq!(back.dimensions(), (3, 2));
        assert_eq!(back.get_pixel(2, 1).0, [1, 2, 3, 255]);
    }
}
