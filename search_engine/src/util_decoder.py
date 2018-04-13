#!/usr/bin/env python
# -*- coding: utf-8 -*-
# (c) Copyright 2016 Kevin McGuinness. All Rights Reserved.
"""
Utilities
"""
import cv2
import numpy as np
import requests

from base64 import b64decode, b64encode


__author__ = "Kevin McGuinness"
__version__ = "1.0"


supported_mimetypes = {
    'image/png': '.png',
    'image/jpeg': '.jpg',
    'image/bmp': '.bmp',
    'image/gif': '.gif',
}


class DecodeError(Exception):
    pass


def parse_data_url(data_url):
    """
    Parse a data url into a tuple of params and the encoded data.

    E.g.
    >>> data_url = "data:image/png;base64,ABC123xxx"
    >>> params, encoded_data = parse_data_url(data_url)
    >>> params
    ('image/png', 'base64')
    >>> data
    'ABC123xxx'

    """
    # e.g. data:image/png;base64,xxx..
    if not data_url.startswith('data:'):
        raise ValueError('not a data url')
    data_url = data_url[5:]
    params, data = data_url.split(',')
    params = params.split(';')
    return params, data


def get_image_data_and_extension_from_data_url(data_url):
    """
    Parse image data encoded in a data URL and return the decoded (raw) data
    and an appropriate file extension to use.
    """
    params, data = parse_data_url(data_url)
    if len(params) < 2:
        raise ValueError('invalid data url: not enough params')
    mimetype = params[0]
    encoding = params[-1]
    if encoding != 'base64':
        raise ValueError('Unsupported encoding: {}'.format(encoding))
    if mimetype not in supported_mimetypes:
        raise ValueError('Unsupported mimetype: {}'.format(mimetype))
    data = b64decode(data)
    extension = supported_mimetypes[mimetype]
    return data, extension


def decode_image(data, flags=-1):
    arr = np.fromstring(data, dtype=np.uint8)
    im = cv2.imdecode(arr, flags)
    if im is None:
        raise DecodeError('Unable to decode image')
    return from_opencv_format(im)


def decode_image_data_url(data_url, flags=-1):
    data, ext = get_image_data_and_extension_from_data_url(data_url)
    return decode_image(data, flags)


def from_opencv_format(image):
    if image.ndim == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    return image


def to_opencv_format(image):
    if image.ndim == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    return image


def encode_image_as_data_url(image, format='png'):
    if format == 'png':
        prefix = 'data:image/png;base64,'
    else:
        prefix = 'data:image/jpeg;base64,'
    opencv_image = to_opencv_format(image)
    retval, encoded_image = cv2.imencode("." + format, opencv_image)
    encoded_image = b64encode(encoded_image)
    return prefix + encoded_image


def download_image(url, flags=-1):
    r = requests.get(url)
    r.raise_for_status()
    return decode_image(r.content, flags)
