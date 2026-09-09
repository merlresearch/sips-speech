# Copyright (c) 2026 Mitsubishi Electric Research Laboratories (MERL)
# SPDX-License-Identifier: AGPL-3.0-or-later

from sips_speech.utils.config import call_func, construct_class


def _capture_kwargs(**kwargs):
    return kwargs


class _CaptureInitKwargs:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_call_func_allows_func_name_keyword_for_called_function():
    assert call_func(_capture_kwargs, func_name="inner") == {"func_name": "inner"}


def test_call_func_supports_legacy_keyword_target():
    assert call_func(func_name=lambda value: value + 1, value=2) == 3


def test_construct_class_allows_class_name_keyword_for_constructed_class():
    obj = construct_class(_CaptureInitKwargs, class_name="inner")

    assert obj.kwargs == {"class_name": "inner"}


def test_construct_class_allows_func_name_keyword_for_constructed_class():
    obj = construct_class(_CaptureInitKwargs, func_name="inner")

    assert obj.kwargs == {"func_name": "inner"}


def test_construct_class_supports_legacy_keyword_target():
    obj = construct_class(class_name=_CaptureInitKwargs, value=2)

    assert obj.kwargs == {"value": 2}
