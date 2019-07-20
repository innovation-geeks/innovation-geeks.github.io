#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Update (or insert) sitemap:lastmod of posts by their git log date.

Envirnment:
  - git
  - ruamel.yaml
"""

import glob
import os
import subprocess
import shutil

from utils.frontmatter_getter import get_yaml
from ruamel.yaml import YAML

POSTS_PATH = "_posts"


def main():
    count = 0
    yaml = YAML()

    for post in glob.glob(os.path.join(POSTS_PATH, "*.md")):
        git_lastmod = subprocess.check_output([
            "git", "log", "-1", "--pretty=%ad", "--date=short", post]).strip()

        if not git_lastmod:
            continue

        frontmatter, line_num = get_yaml(post)
        meta = yaml.load(frontmatter)

        if 'sitemap' in meta:
            if 'lastmod' in meta['sitemap']:
                if meta['sitemap']['lastmod'] == git_lastmod:
                    continue

                meta['sitemap']['lastmod'] = git_lastmod

            else:
                meta['sitemap'].insert(0, 'lastmod', git_lastmod)
        else:
            meta.insert(line_num, 'sitemap', {'lastmod': git_lastmod})

        output = 'new.md'
        if os.path.isfile(output):
            os.remove(output)

        with open(output, 'w') as new, open(post, 'r') as old:
            new.write("---\n")
            yaml.dump(meta, new)
            new.write("---\n")
            line_num += 2

            lines = old.readlines()

            for line in lines:
                if line_num > 0:
                    line_num -= 1
                    continue
                else:
                    new.write(line)

        shutil.move(output, post)
        count += 1
        print("[INFO] update 'lastmod' for: '{}'".format(post))

    print("[NOTICE] Success! Update all posts's lastmod.\n")

    if count > 0:
        subprocess.call(["git", "add", POSTS_PATH])
        subprocess.call(["git", "commit", "-m",
                         "[Automation] Update lastmod of post(s)."])


main()
