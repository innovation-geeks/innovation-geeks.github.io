---
title: jupyter-notebook-tips
date: 2020-06-08 09:45:00 +0800
categories: [python]
tags: [jupyter]
comments: true
toc: true
sitemap:
  lastmod: !!binary |
    MjAyMC0wNi0wOA==
---

# 输出格式转换

转换为markdown格式：

```shell
jupyter nbconvert --to markdown solution.ipynb
```

转换为pdf格式：

```shell
jupyter nbconvert --to pdf solution.ipynb
```

其他支持格式列表：

```python
['asciidoc', 'custom', 
'html', 'html_ch', 'html_embed', 'html_toc', 'html_with_lenvs', 'html_with_toclenvs',
'latex', 'latex_with_lenvs', 
'markdown', 
'notebook', 
'pdf', 
'python', 'rst', 'script', 
'selectLanguage',
'slides', 'slides_with_lenvs']
```

查看所有支持选项：

```shell
jupyter nbconvert -h
```