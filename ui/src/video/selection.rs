//! Director ordered selection: one primary across Scene, Section and audio.
use super::*;
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Item {
    Scene(SceneId),
    Section(SectionId),
    Audio(AudioClipId),
}
impl Item {
    fn kind(&self) -> u8 {
        match self {
            Self::Scene(_) => 0,
            Self::Section(_) => 1,
            Self::Audio(_) => 2,
        }
    }
}
fn ordered(project: &VideoProject, kind: u8) -> Vec<Item> {
    let mut values = match kind {
        0 => project
            .scenes
            .iter()
            .map(|s| {
                (
                    editing::seconds(s.start),
                    0,
                    editing::seconds(s.start) + editing::seconds(s.duration),
                    Item::Scene(s.id.clone()),
                )
            })
            .collect::<Vec<_>>(),
        1 => project
            .scenes
            .iter()
            .flat_map(|s| {
                s.sections.iter().map(move |v| {
                    let start = editing::seconds(s.start) + editing::seconds(v.start);
                    (
                        start,
                        0,
                        start + editing::seconds(v.duration),
                        Item::Section(v.id.clone()),
                    )
                })
            })
            .collect(),
        _ => project
            .audio_tracks
            .iter()
            .flat_map(|t| &t.clips)
            .map(|c| {
                let start = editing::seconds(c.start);
                (
                    start,
                    c.lane,
                    start + editing::seconds(c.duration),
                    Item::Audio(c.id.clone()),
                )
            })
            .collect(),
    };
    values.sort_by(|a, b| {
        a.0.total_cmp(&b.0)
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| a.2.total_cmp(&b.2))
    });
    values.into_iter().map(|v| v.3).collect()
}
impl Timeline {
    pub(super) fn selection_items(&self, p: &VideoProject) -> Vec<Item> {
        if !self.selection_order.is_empty() {
            return self
                .selection_order
                .iter()
                .filter(|item| ordered(p, item.kind()).contains(item))
                .cloned()
                .collect();
        }
        let mut items = self
            .section_selection
            .iter()
            .cloned()
            .map(Item::Section)
            .collect::<Vec<_>>();
        for id in &self.audio_selection_order {
            if self.audio_selected.contains(id) {
                items.push(Item::Audio(id.clone()));
            }
        }
        for id in &self.audio_selected {
            let item = Item::Audio(id.clone());
            if !items.contains(&item) {
                items.push(item);
            }
        }
        if items.is_empty() {
            items.extend(self.selected.iter().cloned().map(Item::Scene));
        }
        let primary = self
            .section
            .clone()
            .map(Item::Section)
            .or_else(|| self.audio_primary.clone().map(Item::Audio));
        if let Some(primary) = primary
            && items.contains(&primary)
        {
            items.retain(|v| v != &primary);
            items.push(primary);
        }
        items
    }
    pub(super) fn primary_scene(&self, p: &VideoProject) -> Option<SceneId> {
        match self.selection_items(p).last()? {
            Item::Scene(id) => Some(id.clone()),
            Item::Section(id) => p
                .scenes
                .iter()
                .find(|s| s.sections.iter().any(|v| &v.id == id))
                .map(|s| s.id.clone()),
            Item::Audio(_) => None,
        }
    }
    pub(super) fn pick_item(
        &mut self,
        p: &VideoProject,
        item: Item,
        toggle: bool,
        range: bool,
        context: bool,
    ) {
        let current = self.selection_items(p);
        let next = if context && current.contains(&item) {
            current
        } else if range {
            choose(&current, &ordered(p, item.kind()), &item, false, true)
        } else {
            let all = (0..3).flat_map(|kind| ordered(p, kind)).collect::<Vec<_>>();
            choose(&current, &all, &item, toggle, false)
        };
        self.set_selection(p, next);
    }
    /// Nothing selected, so the inspector shows the project.
    pub(super) fn clear_selection(&mut self, p: &VideoProject) {
        self.boundary = false;
        self.selected_take = None;
        self.set_selection(p, vec![]);
    }
    pub(super) fn set_selection(&mut self, p: &VideoProject, items: Vec<Item>) {
        self.selected.clear();
        self.section_selection.clear();
        self.clear_audio_selection();
        for item in &items {
            match item {
                Item::Scene(id) => {
                    self.selected.insert(id.clone());
                }
                Item::Section(id) => {
                    self.section_selection.push(id.clone());
                    if let Some(owner) = p
                        .scenes
                        .iter()
                        .find(|s| s.sections.iter().any(|v| &v.id == id))
                    {
                        self.selected.insert(owner.id.clone());
                    }
                }
                Item::Audio(id) => {
                    self.audio_selected.insert(id.clone());
                    self.audio_selection_order.push(id.clone());
                }
            }
        }
        self.section = match items.last() {
            Some(Item::Section(id)) => Some(id.clone()),
            _ => None,
        };
        self.audio_primary = match items.last() {
            Some(Item::Audio(id)) => Some(id.clone()),
            _ => None,
        };
        self.selection_order = items;
        self.boundary = false;
        self.sync_draft();
    }
    pub(super) fn reconcile_selection(&mut self) {
        let primary_matches = match self.selection_order.last() {
            Some(Item::Section(id)) => self.section.as_ref() == Some(id),
            Some(Item::Audio(id)) => self.audio_primary.as_ref() == Some(id),
            Some(Item::Scene(id)) => {
                self.section.is_none() && self.audio_primary.is_none() && self.selected.contains(id)
            }
            None => true,
        };
        let sections = self
            .selection_order
            .iter()
            .filter_map(|i| match i {
                Item::Section(id) => Some(id.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        let audio = self
            .selection_order
            .iter()
            .filter_map(|i| match i {
                Item::Audio(id) => Some(id.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        let mut scenes = self
            .selection_order
            .iter()
            .filter_map(|i| match i {
                Item::Scene(id) => Some(id.clone()),
                _ => None,
            })
            .collect::<BTreeSet<_>>();
        if let Some(p) = self.document.as_ref().map(|d| &d.project) {
            for id in &sections {
                if let Some(s) = p
                    .scenes
                    .iter()
                    .find(|s| s.sections.iter().any(|v| &v.id == id))
                {
                    scenes.insert(s.id.clone());
                }
            }
        }
        let projection_matches = self.selection_order.is_empty()
            || (sections == self.section_selection
                && audio == self.audio_selection_order
                && audio.iter().cloned().collect::<BTreeSet<_>>() == self.audio_selected
                && scenes == self.selected);
        if !primary_matches || !projection_matches {
            self.selection_order.clear();
        }
    }
}
pub(super) fn choose<T: Clone + PartialEq>(
    current: &[T],
    ordered: &[T],
    target: &T,
    toggle: bool,
    range: bool,
) -> Vec<T> {
    if range {
        let anchor = current
            .iter()
            .rev()
            .find(|id| ordered.contains(id))
            .unwrap_or(target);
        if let (Some(a), Some(b)) = (
            ordered.iter().position(|id| id == anchor),
            ordered.iter().position(|id| id == target),
        ) {
            let mut result = ordered[a.min(b)..=a.max(b)].to_vec();
            result.retain(|id| id != target);
            result.push(target.clone());
            return result;
        }
    }
    if !toggle {
        return vec![target.clone()];
    }
    let mut result = current
        .iter()
        .filter(|id| ordered.contains(id))
        .cloned()
        .collect::<Vec<_>>();
    if result.contains(target) {
        result.retain(|id| id != target);
    } else {
        result.push(target.clone());
    }
    result
}
#[cfg(test)]
mod tests {
    use super::*;
    fn timeline() -> (Timeline, VideoProject) {
        let mut p = super::super::media_picker::tests::project();
        let mut second = p.scenes[0].clone();
        second.id = SceneId("second".into());
        second.start = editing::time(3.);
        second.sections[0].id = SectionId("section_b".into());
        p.scenes.push(second);
        let mut bare = p.scenes[0].clone();
        bare.id = SceneId("bare".into());
        bare.start = editing::time(6.);
        bare.sections.clear();
        p.scenes.push(bare);
        p.audio_tracks = vec![AudioTrack {
            id: AudioTrackId("own_track".into()),
            clips: (0..3)
                .map(|lane| {
                    let mut c = AudioClip::new(
                        AudioClipId(format!("audio_{lane}")),
                        editing::time(0.),
                        editing::time(f64::from(5 - lane)),
                        &p.audio,
                    );
                    c.lane = lane;
                    c
                })
                .collect(),
        }];
        let mut t = Timeline::default();
        t.document = Some(Snapshot::with(p.clone(), json!({})));
        (t, p)
    }
    #[test]
    fn mixed_toggle_primary_context_and_shift_follow_one_order() {
        let (mut t, p) = timeline();
        let a = Item::Section(p.scenes[0].sections[0].id.clone());
        let b = Item::Section(SectionId("section_b".into()));
        let audio = Item::Audio(AudioClipId("audio_0".into()));
        let scene = Item::Scene(SceneId("bare".into()));
        t.pick_item(&p, a.clone(), false, false, false);
        t.pick_item(&p, scene.clone(), true, false, false);
        t.pick_item(&p, audio.clone(), true, false, false);
        t.pick_item(&p, b.clone(), true, false, false);
        t.pick_item(&p, b.clone(), true, false, false);
        assert_eq!(t.selection_items(&p), [a.clone(), scene, audio.clone()]);
        assert_eq!(t.audio_primary, Some(AudioClipId("audio_0".into())));
        assert!(t.section.is_none());
        assert!(t.scene().is_none());
        t.pick_item(&p, a.clone(), false, false, true);
        assert_eq!(t.selection_items(&p).last(), Some(&audio));
        t.pick_item(&p, b.clone(), true, true, false);
        assert_eq!(t.selection_items(&p), [a, b]);
        assert!(t.audio_selected.is_empty());
        t.pick_item(&p, audio.clone(), false, false, true);
        assert_eq!(t.selection_items(&p), [audio]);
    }
    #[test]
    fn audio_shift_orders_same_start_by_lane_before_end() {
        let (mut t, p) = timeline();
        t.pick_item(
            &p,
            Item::Audio(AudioClipId("audio_2".into())),
            false,
            false,
            false,
        );
        t.pick_item(
            &p,
            Item::Audio(AudioClipId("audio_0".into())),
            false,
            true,
            false,
        );
        assert_eq!(
            t.selection_items(&p),
            [1, 2, 0].map(|i| Item::Audio(AudioClipId(format!("audio_{i}"))))
        );
    }
    #[test]
    fn toggles_and_ranges_keep_the_clicked_item_primary_and_ignore_stale_ids() {
        let ordered = ["a", "b", "c", "d"];
        assert_eq!(choose(&["a"], &ordered, &"c", true, false), ["a", "c"]);
        assert_eq!(choose(&["a", "c"], &ordered, &"a", true, false), ["c"]);
        assert_eq!(choose(&["d"], &ordered, &"b", false, true), ["c", "d", "b"]);
        assert_eq!(choose(&["missing"], &ordered, &"b", false, true), ["b"]);
        assert_eq!(choose(&["a", "c"], &ordered, &"b", false, false), ["b"]);
    }
    #[test]
    fn legacy_post_action_collapsing_same_primary_discards_old_group_order() {
        let (mut t, p) = timeline();
        let a = Item::Section(p.scenes[0].sections[0].id.clone());
        let b = SectionId("section_b".into());
        t.pick_item(&p, a, false, false, false);
        t.pick_item(&p, Item::Section(b.clone()), true, false, false);
        t.selected = BTreeSet::from([SceneId("second".into())]);
        t.section_selection = vec![b.clone()];
        t.section = Some(b.clone());
        t.sync_draft();
        assert_eq!(t.selection_items(&p), [Item::Section(b)]);
        assert_eq!(t.primary_scene(&p), Some(SceneId("second".into())));
    }
}
