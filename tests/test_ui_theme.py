"""Theme and copy gates: contrast math, focus visibility, honest wording.

These pin the antislop decisions so a later tweak cannot silently reintroduce
a failing pair or a banned character.
"""


def _luminance(hex_color):
    channels = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _ratio(foreground, background):
    lighter = max(_luminance(foreground), _luminance(background))
    darker = min(_luminance(foreground), _luminance(background))
    return (lighter + 0.05) / (darker + 0.05)


def test_text_pairs_meet_wcag_aa():
    from src.gui import shell
    # Body and hint text on every surface it sits on.
    assert _ratio(shell.TEXT_MUTED, shell.BG) >= 4.5
    assert _ratio(shell.TEXT_MUTED, shell.CARD) >= 4.5
    assert _ratio(shell.TEXT_MUTED, shell.PILL) >= 4.5
    assert _ratio(shell.TEXT, shell.BG) >= 4.5
    assert _ratio(shell.TEXT, shell.CARD) >= 4.5
    # Reversed pairs on filled controls.
    assert _ratio(shell.PRIMARY_TEXT, shell.PRIMARY) >= 4.5
    assert _ratio('#ffffff', shell.DANGER) >= 4.5
    assert _ratio(shell.WARN, shell.CARD) >= 4.5
    assert _ratio(shell.WARN, shell.BG) >= 4.5


def test_live_accent_is_a_single_color():
    from src.gui import shell
    assert shell.LIVE.startswith('#') and shell.LIVE != shell.PRIMARY
    assert _ratio(shell.LIVE, shell.BG) >= 3.0
    assert _ratio(shell.LIVE, shell.CARD) >= 3.0


def test_keyboard_focus_stays_visible_without_moving_layout():
    from src.gui.shell import stylesheet
    css = stylesheet('assets')
    for selector in ('QPushButton:focus', 'QComboBox:focus', 'QLineEdit:focus',
                     'QCheckBox:focus', 'QSlider:focus'):
        assert selector in css, selector


def test_guide_copy_has_no_em_dash_or_buzzwords():
    from src.gui.tutorial import PAGES
    banned = ('—', 'シームレス', '次世代', '革新的', '最先端', 'パワフル',
              'AI Powered', 'Seamless', '革命的')
    assert len(PAGES) == 6
    for spec in PAGES:
        for word in banned:
            assert word not in spec['body'], (spec['title'], word)


def test_status_pill_marks_real_states_only():
    from src.gui.shell import PageHeader
    assert set(PageHeader.BADGES) == {'stopped', 'running', 'error'}
