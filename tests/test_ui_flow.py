"""The window shows only controls that can act, and each visible one does something.

The shipped standard voice is a MeanVC2 registration: its pitch, EQ, coding chunk and
prosody are fixed, so those controls are hidden instead of left greyed out. These tests
pin that behaviour, the home-page quick voice picker, and the preset-rail link.
"""
import pytest

from tests.test_ai_gui import ai_window, pump  # noqa: F401  (ai_window is used as a fixture)
from src.gui.tutorial import PAGES


def test_home_page_is_a_numbered_step_flow(ai_window):
    app, window, backend, bridge = ai_window
    pump(app, lambda: window._active_page == 'home')
    # ① devices, ② mode/voice and ③ the transport, all on the page the user starts from.
    assert [number for number, _card in window.home_step_cards] == ['①', '②', '③']
    assert [card.objectName() for _number, card in window.home_step_cards] == \
        ['stepCard', 'stepCard', 'stepCard']
    for _number, card in window.home_step_cards:
        # ③ lives inside the transport footer, so walk up instead of assuming depth.
        parent = card.parentWidget()
        while parent is not None and parent is not window.pages['home'][0]:
            parent = parent.parentWidget()
        assert parent is window.pages['home'][0], card


def test_standard_voice_hides_controls_that_cannot_apply(ai_window):
    app, window, backend, bridge = ai_window
    window._show_page('advanced')
    pump(app, lambda: window._active_page == 'advanced')
    # MeanVC2 fixes the chunk, pitch and brightness; prosody stays off for it.
    for widget in (window.ai_quality, window.ai_brightness, window.ai_pitch, window.ai_wait):
        assert not window._container(widget).isVisible(), widget
    assert window.ai_threads.isVisible()
    assert not window.prosody_tab.isVisible()
    assert window.prosody_fixed_note.isVisible()


def test_control_visibility_follows_the_mode_and_route(ai_window):
    app, window, backend, bridge = ai_window
    # The tuning block lives on the advanced page; only the comparison route uses it.
    window._show_page('advanced')
    pump(app, lambda: window._active_page == 'advanced')
    pump(app, lambda: not window.tune_header.isVisible())
    assert not window._container(window.tune_sib).isVisible()
    window.ai_delivery.setCurrentIndex(window.ai_delivery.findData('utterance_x_natural'))
    pump(app, lambda: window._container(window.tune_sib).isVisible())
    for widget in window._tune_widgets:
        assert window._container(widget).isVisible(), widget
    window.ai_delivery.setCurrentIndex(window.ai_delivery.findData('utterance_lavasr'))
    pump(app, lambda: not window._container(window.tune_sib).isVisible())
    # Committing an unspoken utterance exists only on the utterance route.
    window._show_page('library')
    pump(app, lambda: window._active_page == 'library' and window.finish_phrase.isVisible())
    window.ai_delivery.setCurrentIndex(window.ai_delivery.findData('streaming'))
    pump(app, lambda: not window.finish_phrase.isVisible())
    window.ai_delivery.setCurrentIndex(window.ai_delivery.findData('utterance_lavasr'))
    pump(app, lambda: window.finish_phrase.isVisible())


def test_dsp_page_is_visible_and_drivable_only_on_the_dsp_path(ai_window):
    app, window, backend, bridge = ai_window
    window._show_page('presets')
    pump(app, lambda: window._active_page == 'presets')
    # AI Voice is the active mode, so the page explains itself instead of showing
    # controls that would do nothing.
    pump(app, lambda: not window.pitch_display.isVisible())
    assert window.dsp_notice.text().startswith('いまは「AI Voice」です。')
    assert window.dsp_switch.isVisible()
    window.dsp_switch.click()
    pump(app, lambda: window.mode.currentData() == 'female_dsp'
         and window.pitch_display.isVisible())
    assert window.pitch.isVisible() and window.formant.isVisible()
    assert not window.dsp_switch.isVisible()
    assert window.dsp_notice.text() == ''


def test_quick_voice_picker_and_library_combo_stay_in_step(ai_window):
    app, window, backend, bridge = ai_window
    window._show_page('home')
    pump(app, lambda: window._active_page == 'home')
    assert window.voice_quick.isVisible()
    assert [window.voice_quick.itemData(i) for i in range(window.voice_quick.count())] == \
        [window.ai_model.itemData(i) for i in range(window.ai_model.count())]
    assert window.voice_quick.currentData() == window.ai_model.currentData()
    # Choosing on the library page updates the home picker without re-sending a load.
    window.ai_model.setCurrentIndex(window.ai_model.count() - 1)
    pump(app, lambda: window.voice_quick.currentData() == window.ai_model.currentData())
    window.mode.setCurrentIndex(window.mode.findData('original'))
    pump(app, lambda: not window.voice_quick.isVisible())


def test_preset_rail_link_opens_the_library_page(ai_window):
    app, window, backend, bridge = ai_window
    assert window._active_page == 'home'
    window.rail.link.click()
    pump(app, lambda: window._active_page == 'library')
    assert window.pages['library'][0].isVisible()


def test_guide_pages_point_at_real_widgets(ai_window):
    app, window, backend, bridge = ai_window
    for spec in PAGES:
        page = spec.get('page')
        assert page in window.pages, spec
        target = spec.get('target')
        if target:
            widget = getattr(window, target, None)
            assert widget is not None, f'{spec["title"]} points at {target}'
            # The guide must never point outside the page it just opened.
            parent = widget
            while parent is not None and parent is not window.pages[page][0]:
                parent = parent.parentWidget()
            assert parent is window.pages[page][0], f'{target} is not on {page}'
