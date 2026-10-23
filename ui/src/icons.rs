//! Small outline icons, drawn at the same logical size on every display scale.
use eframe::egui::{self, Color32, Painter, Rect, Stroke};

#[derive(Clone, Copy, Debug)]
pub enum Icon {
    Image,
    Save,
    Edit,
    Enhance,
    Duplicate,
    Copy,
    Insert,
    Lora,
    CharacterSheet,
    Video,
    Audio,
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
    Check,
    Undo,
    Redo,
    Compare,
    Swap,
    Unlink,
    Animate,
    Play,
    Pause,
    Story,
    Montage,
    Review,
    Link,
    Menu,
    Person,
    Library,
    Import,
    Export,
    Download,
    Back,
    Lock,
    Info,
    Scene,
    PanelRight,
    PanelBottom,
    Snap,
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
    let triangle = |points: [[f32; 2]; 3]| {
        painter.add(egui::Shape::closed_line(
            points.iter().map(|[x, y]| pos(*x, *y)).collect(),
            stroke,
        ));
    };
    match icon {
        Icon::Animate => {
            box_at(2., 5., 20., 14., 2);
            for x in [6., 18.] {
                line(&[[x, 5.], [x, 19.]]);
            }
            triangle([[10., 9.], [10., 15.], [15., 12.]]);
        }
        Icon::Play => triangle([[7., 4.], [7., 20.], [19., 12.]]),
        Icon::Pause => {
            box_at(6., 4., 4., 16., 1);
            box_at(14., 4., 4., 16., 1);
        }
        Icon::Story => {
            line(&[
                [12., 6.],
                [8., 4.],
                [2., 4.],
                [2., 19.],
                [8., 19.],
                [12., 21.],
            ]);
            line(&[
                [12., 6.],
                [16., 4.],
                [22., 4.],
                [22., 19.],
                [16., 19.],
                [12., 21.],
            ]);
            line(&[[12., 6.], [12., 21.]]);
        }
        Icon::Montage => {
            for x in [2., 9., 16.] {
                box_at(x, 7., 6., 10., 1);
            }
            line(&[[2., 20.], [22., 20.]]);
            line(&[[19., 18.], [22., 20.], [19., 22.]]);
        }
        Icon::Review => {
            box_at(4., 4., 16., 18., 2);
            box_at(9., 2., 6., 4., 1);
            line(&[[8., 14.], [11., 17.], [16., 10.]]);
        }
        Icon::Link => {
            line(&[
                [10., 14.],
                [7., 17.],
                [5., 17.],
                [3., 15.],
                [3., 13.],
                [7., 9.],
                [9., 9.],
                [11., 11.],
            ]);
            line(&[
                [14., 10.],
                [17., 7.],
                [19., 7.],
                [21., 9.],
                [21., 11.],
                [17., 15.],
                [15., 15.],
                [13., 13.],
            ]);
            line(&[[9., 15.], [15., 9.]]);
        }
        Icon::Menu => {
            for y in [6., 12., 18.] {
                line(&[[4., y], [20., y]]);
            }
        }
        Icon::Person => {
            painter.circle_stroke(pos(12., 8.), 4. * s, stroke);
            line(&[
                [4., 21.],
                [5., 16.],
                [9., 14.],
                [15., 14.],
                [19., 16.],
                [20., 21.],
            ]);
        }
        Icon::Library => {
            box_at(3., 4., 4., 17., 1);
            box_at(9., 4., 4., 17., 1);
            line(&[[15., 6.], [19., 5.], [22., 20.], [18., 21.], [15., 6.]]);
        }
        Icon::Import | Icon::Export | Icon::Download => {
            line(&[[3., 15.], [3., 21.], [21., 21.], [21., 15.]]);
            line(&[[12., 3.], [12., 16.]]);
            if matches!(icon, Icon::Export) {
                line(&[[7., 8.], [12., 3.], [17., 8.]]);
            } else {
                line(&[[7., 11.], [12., 16.], [17., 11.]]);
            }
            if matches!(icon, Icon::Download) {
                line(&[[8., 18.], [16., 18.]]);
            }
        }
        Icon::Back => {
            line(&[[20., 12.], [4., 12.]]);
            line(&[[11., 5.], [4., 12.], [11., 19.]]);
        }
        Icon::Lock => {
            box_at(4., 10., 16., 12., 2);
            line(&[
                [8., 10.],
                [8., 6.],
                [10., 3.],
                [14., 3.],
                [16., 6.],
                [16., 10.],
            ]);
        }
        Icon::Scene => {
            box_at(3., 9., 18., 12., 2);
            line(&[[3., 9.], [4., 4.], [20., 2.], [21., 7.], [3., 9.]]);
            line(&[[9., 3.5], [11., 8.]]);
            line(&[[15., 2.8], [17., 7.5]]);
        }
        Icon::PanelRight => {
            box_at(3., 4., 18., 16., 2);
            line(&[[15., 4.], [15., 20.]]);
        }
        Icon::Snap => {
            // A magnet: the timeline edge an item jumps to.
            line(&[[6., 3.], [6., 11.]]);
            line(&[[18., 3.], [18., 11.]]);
            painter.add(egui::Shape::line(
                (0..=12)
                    .map(|i| {
                        let a = std::f32::consts::PI * i as f32 / 12.;
                        pos(12. - 6. * a.cos(), 11. + 6. * a.sin())
                    })
                    .collect(),
                stroke,
            ));
            line(&[[6., 6.], [10., 6.]]);
            line(&[[14., 6.], [18., 6.]]);
        }
        Icon::PanelBottom => {
            box_at(3., 4., 18., 16., 2);
            line(&[[3., 14.], [21., 14.]]);
        }
        Icon::Info => {
            painter.circle_stroke(pos(12., 12.), 9.5 * s, stroke);
            line(&[[12., 11.], [12., 17.]]);
            painter.circle_filled(pos(12., 7.5), 1.1 * s, color);
        }
        Icon::Unlink => {
            line(&[
                [9., 6.],
                [7., 6.],
                [4., 9.],
                [4., 13.],
                [7., 16.],
                [9., 16.],
            ]);
            line(&[
                [15., 8.],
                [17., 8.],
                [20., 11.],
                [20., 15.],
                [17., 18.],
                [15., 18.],
            ]);
            line(&[[13., 3.], [11., 8.]]);
            line(&[[13., 16.], [11., 21.]]);
        }
        Icon::Compare => {
            box_at(3., 4., 18., 16., 2);
            line(&[[12., 4.], [12., 20.]]);
            line(&[[6., 10.], [9., 10.]]);
            line(&[[15., 14.], [18., 14.]]);
        }
        Icon::Swap => {
            line(&[[4., 8.], [20., 8.], [16., 4.]]);
            line(&[[20., 16.], [4., 16.], [8., 20.]]);
        }
        Icon::Audio => {
            for (x, a, b) in [
                (4., 9., 15.),
                (8., 5., 19.),
                (12., 2., 22.),
                (16., 6., 18.),
                (20., 9., 15.),
            ] {
                line(&[[x, a], [x, b]]);
            }
        }
        Icon::Undo => {
            line(&[[9., 4.], [4., 9.], [9., 14.]]);
            line(&[[4., 9.], [14., 9.], [19., 13.], [19., 20.]]);
        }
        Icon::Redo => {
            line(&[[15., 4.], [20., 9.], [15., 14.]]);
            line(&[[20., 9.], [10., 9.], [5., 13.], [5., 20.]]);
        }
        Icon::Enhance => {
            line(&[[4., 20.], [16., 8.], [19., 11.], [7., 23.], [4., 20.]]);
            line(&[[5., 3.], [5., 9.]]);
            line(&[[2., 6.], [8., 6.]]);
            line(&[[18., 1.], [18., 5.]]);
            line(&[[16., 3.], [20., 3.]]);
        }
        Icon::Insert => {
            line(&[[4., 5.], [20., 5.]]);
            line(&[[4., 10.], [12., 10.]]);
            line(&[[4., 20.], [20., 20.]]);
            line(&[[17., 9.], [17., 16.], [13., 12.]]);
            line(&[[17., 16.], [21., 12.]]);
        }
        Icon::Duplicate | Icon::Copy => {
            box_at(8., 8., 13., 13., 2);
            line(&[[5., 16.], [3., 16.], [3., 3.], [16., 3.], [16., 5.]]);
        }
        Icon::Lora => {
            line(&[[3., 7.], [12., 3.], [21., 7.], [12., 11.], [3., 7.]]);
            line(&[[3., 12.], [12., 16.], [21., 12.]]);
            line(&[[3., 17.], [12., 21.], [21., 17.]]);
        }
        Icon::Edit => {
            line(&[
                [4., 16.],
                [16., 4.],
                [20., 8.],
                [8., 20.],
                [3., 21.],
                [4., 16.],
            ]);
            line(&[[13., 7.], [17., 11.]]);
        }
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
        Icon::Check => line(&[[4., 12.], [9., 17.], [20., 6.]]),
        Icon::Save => {
            box_at(4., 3., 16., 18., 1);
            box_at(7., 3., 9., 6., 0);
            box_at(7., 13., 10., 8., 0);
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
