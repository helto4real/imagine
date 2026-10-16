//! Small outline icons, drawn at the same logical size on every display scale.
use eframe::egui::{self, Color32, Painter, Rect, Stroke};

#[derive(Clone, Copy)]
pub enum Icon {
    Image,
    CharacterSheet,
    Video,
    Settings,
    Eye,
    EyeOff,
    Plus,
    Up,
    Down,
    Trash,
    Fit,
    Actual,
    Hand,
    Generate,
    Upscale,
    Retry,
    Stop,
    Folder,
    Close,
}

pub fn paint(painter: &Painter, rect: Rect, icon: Icon, color: Color32) {
    let s = rect.width().min(rect.height()) / 24.;
    let origin = rect.center() - egui::vec2(12., 12.) * s;
    let pos = |x, y| origin + egui::vec2(x, y) * s;
    let stroke = Stroke::new(1.65 * s, color);
    let line = |points: &[[f32; 2]]| {
        painter.add(egui::Shape::line(
            points.iter().map(|[x, y]| pos(*x, *y)).collect(),
            stroke,
        ));
    };
    let box_at = |x, y, w, h, radius| {
        painter.rect_stroke(
            Rect::from_min_size(pos(x, y), egui::vec2(w, h) * s),
            radius,
            stroke,
            egui::StrokeKind::Middle,
        );
    };
    match icon {
        Icon::Image => {
            box_at(3., 3., 18., 18., 2);
            painter.circle_stroke(pos(8., 8.), 1.5 * s, stroke);
            line(&[[3., 17.], [9., 11.], [13., 15.], [16., 12.], [21., 17.]]);
        }
        Icon::CharacterSheet => {
            box_at(2., 3., 20., 18., 2);
            line(&[[11., 3.], [11., 21.]]);
            painter.circle_stroke(pos(6.5, 9.), 2. * s, stroke);
            line(&[[4., 17.], [4., 14.], [9., 14.], [9., 17.]]);
            for y in [8., 12., 16.] {
                line(&[[14., y], [19., y]]);
            }
        }
        Icon::Upscale => {
            box_at(3., 12., 9., 9., 1);
            line(&[[13., 3.], [21., 3.], [21., 11.]]);
            line(&[[10., 14.], [21., 3.]]);
        }
        Icon::Video => {
            box_at(3., 3., 18., 18., 2);
            for x in [7., 17.] {
                line(&[[x, 3.], [x, 21.]]);
            }
            for y in [7., 12., 17.] {
                line(&[[3., y], [7., y]]);
                line(&[[17., y], [21., y]]);
            }
        }
        Icon::Settings => {
            let mut points = Vec::new();
            for i in 0..32 {
                let angle = i as f32 * std::f32::consts::TAU / 32.;
                let radius = if i % 4 < 2 { 9.5 } else { 7.5 };
                points.push(pos(12. + angle.cos() * radius, 12. + angle.sin() * radius));
            }
            painter.add(egui::Shape::closed_line(points, stroke));
            painter.circle_stroke(pos(12., 12.), 3. * s, stroke);
        }
        Icon::Eye | Icon::EyeOff => {
            line(&[
                [2., 12.],
                [5., 7.],
                [9., 5.],
                [15., 5.],
                [19., 7.],
                [22., 12.],
                [19., 17.],
                [15., 19.],
                [9., 19.],
                [5., 17.],
                [2., 12.],
            ]);
            painter.circle_stroke(pos(12., 12.), 3. * s, stroke);
            if matches!(icon, Icon::EyeOff) {
                line(&[[3., 3.], [21., 21.]]);
            }
        }
        Icon::Plus => {
            line(&[[12., 4.], [12., 20.]]);
            line(&[[4., 12.], [20., 12.]]);
        }
        Icon::Up | Icon::Down => {
            let y = |v| {
                if matches!(icon, Icon::Down) {
                    24. - v
                } else {
                    v
                }
            };
            line(&[[12., y(20.)], [12., y(4.)]]);
            line(&[[5., y(11.)], [12., y(4.)], [19., y(11.)]]);
        }
        Icon::Trash => {
            line(&[[4., 6.], [20., 6.]]);
            line(&[[9., 6.], [9., 3.], [15., 3.], [15., 6.]]);
            line(&[[6., 6.], [7., 21.], [17., 21.], [18., 6.]]);
            line(&[[10., 10.], [10., 17.]]);
            line(&[[14., 10.], [14., 17.]]);
        }
        Icon::Fit => {
            for (x, y, dx, dy) in [
                (3., 3., 1., 1.),
                (21., 3., -1., 1.),
                (3., 21., 1., -1.),
                (21., 21., -1., -1.),
            ] {
                line(&[[x, y + dy * 6.], [x, y], [x + dx * 6., y]]);
                line(&[[x, y], [x + dx * 7., y + dy * 7.]]);
            }
        }
        Icon::Actual => {
            box_at(2., 3., 20., 18., 1);
            line(&[[6., 10.], [8., 8.], [8., 16.]]);
            line(&[[16., 10.], [18., 8.], [18., 16.]]);
            for y in [10., 14.] {
                painter.circle_filled(pos(12.5, y), 0.8 * s, color);
            }
        }
        Icon::Hand => {
            line(&[
                [6., 13.],
                [6., 7.],
                [8., 6.],
                [9., 7.],
                [9., 12.],
                [9., 3.],
                [11., 2.],
                [13., 3.],
                [13., 12.],
                [13., 4.],
                [15., 3.],
                [17., 4.],
                [17., 13.],
                [17., 7.],
                [19., 6.],
                [21., 8.],
                [20., 17.],
                [17., 22.],
                [10., 22.],
                [6., 18.],
                [2., 12.],
                [3., 10.],
                [5., 11.],
                [8., 15.],
            ]);
        }
        Icon::Generate => {
            for (x, y, r) in [(10., 12., 8.), (19., 5., 3.), (19., 20., 2.)] {
                line(&[
                    [x, y - r],
                    [x + r * 0.3, y - r * 0.3],
                    [x + r, y],
                    [x + r * 0.3, y + r * 0.3],
                    [x, y + r],
                    [x - r * 0.3, y + r * 0.3],
                    [x - r, y],
                    [x - r * 0.3, y - r * 0.3],
                    [x, y - r],
                ]);
            }
        }
        Icon::Retry => {
            line(&[
                [20., 8.],
                [17., 4.],
                [11., 3.],
                [6., 5.],
                [3., 10.],
                [4., 16.],
                [8., 20.],
                [14., 21.],
                [19., 18.],
                [21., 13.],
            ]);
            line(&[[20., 2.], [20., 8.], [14., 8.]]);
        }
        Icon::Stop => {
            painter.rect_filled(Rect::from_min_max(pos(6., 6.), pos(18., 18.)), 2, color);
        }
        Icon::Folder => {
            line(&[
                [3., 7.],
                [3., 4.],
                [10., 4.],
                [13., 7.],
                [21., 7.],
                [21., 21.],
                [3., 21.],
                [3., 7.],
                [21., 7.],
            ]);
        }
        Icon::Close => {
            line(&[[5., 5.], [19., 19.]]);
            line(&[[19., 5.], [5., 19.]]);
        }
    }
}
