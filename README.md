# The Jekyll Template for Cotes' Blog

[![Build Status](https://travis-ci.org/cotes2020/cotes-blog.svg?branch=master)](https://travis-ci.org/cotes2020/cotes-blog)
[![GitHub release](https://img.shields.io/github/release/cotes2020/cotes-blog.svg)](https://github.com/cotes2020/cotes-blog/releases)
[![GitHub license](https://img.shields.io/github/license/cotes2020/cotes-blog.svg)](https://github.com/cotes2020/cotes-blog/blob/master/LICENSE)
[![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu)

My personal Jekyll blog, please check the deployment: [https://innovation-geeks.github.io](https://innovation-geeks.github.io).

## Usage

First run should execute the command:

```bash
bundle install
```

`bundle` will install all plugins we need automatically.

Then boot the site:

```bash
bundle exec jekyll serve
```

Open the brower and visit: [http://127.0.0.1:4000](http://127.0.0.1:4000)


## Configuration

Check files in `_data/`, they are:

* meta.yml
* profile.yml
* settins.yml

Open the files and change the fields inside to your own.


## Writting

There are some things you must know before you write a post:

### TOC

The [Front Matter](https://jekyllrb.com/docs/front-matter/) `toc` is to decide whether to display the **Table of Content** or not, default to `ture`, select `false` to disable.

### Comments

The field `comments` is the switch for `Disqus` comment module, the default value is `true`, to disable it select `false`.

>You have to configuration youer Disqus in file `_data/meta.yml`, override the field `disqus-shortname`

### Categories And Tags

The posts' categories and tag pages are stored in the `categoreis/` and `tags/` directories, respectively, and must correspond to the post one by one.

For example, a post with title `The Beautify Rose` has these Front Matter part:

```
---
title: "The Beautify Rose"
categories: [Plant]
tags: [flower]

 ...

---
```

Then you need to create two new files:

1.Create file `categories/Plant.html`, and fill:

```yaml
---
layout: category
title: Plant        # The title of category page.
category: Plant     # The category name in post
---
```


2.Create file `tags/flower.html`, and fill:

```yaml
---
layout: tag
title: Flower       # The title of tag page.
tag: flower         # The tag name in post.
---
```

If you find this to be time consuming, then please use the script tool `pages_generator.py`.

The python script needs [ruamel.yaml](https://pypi.org/project/ruamel.yaml/), make sure it's installed in your environment, then, run the script:

```bash
python _scripts/tools/pages_generator.py
```

Few seconds later, it will create the Categoreis and Tags HTML files automatically.
