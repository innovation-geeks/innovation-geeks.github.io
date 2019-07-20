#!/usr/bin/env python
# -*- coding: utf-8 -*-


def get_yaml(path):
    """
    Read jekyll posts and return a tuple
    that consist of front-matter and its line number.
    """
    end = False
    yaml = ""
    num = 0

    with open(path, 'r') as f:

        for line in f.readlines():
            if line.strip() == '---':
                if end:
                    break
                else:
                    end = True
                    continue
            else:
                num += 1

            yaml += line

    return yaml, num
