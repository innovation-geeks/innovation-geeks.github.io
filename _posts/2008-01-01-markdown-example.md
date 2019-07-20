---
title: Markdown Example
date: 2008-01-01 00:00:00 +0800
categories: [tutorial]
tags: [markdown]
comments: true
toc: true
sitemap:
  lastmod: !!binary |
    MjAxOS0wNy0yMA==
---

## Typograpy

# H1

## H2

### H3

#### H4

##### H5

## Tables

|Company|Contact|Country|
|:---|:--|---:|
|Alfreds Futterkiste | Maria Anders | Germany
|Island Trading | Helen Bennett | UK
|Magazzini Alimentari Riuniti | Giovanni Rovelli | Italy


## Code Snippet

### Code hightlight

This is a `sample` to `code hightlight`.

### liquid raw

{% highlight liquid %}
{% raw %}
{% highlight html %}
<p>This is some text in a paragraph.</p>
{% endhighlight %}
{% endraw %}
{% endhighlight %}

### XML
{% capture _code %}
{% highlight xml linenos %}
<navMap>
  <navPoint playOrder="1" id="toc-1">
    <navLabel>
      <text>Vol_26</text>
        <text>No.375 比达、杜拉格斯出发</text>
      </navLabel>
      <content src="Text/part0007.xhtml" />
    </navPoint>
  </navPoint>
</navMap>
{% endhighlight%}
{% endcapture %}{% include fixlinenos.html %}{{ _code }}


### HTML

{% highlight html %}
{% raw %}
{% highlight html linenos %}
<p>{% if site.data.showAuthor %}Author:{{ site.data.author }}{% endif %}</p>
<p>This is some text in a paragraph.</p>
{% endhighlight %}
{% endraw %}
{% endhighlight %}

#### html-linenos

* No-Scrolling

{% capture _code %}
{% highlight html linenos %}
{% raw %}
<div class="panel-group">
  <div class="panel panel-default">
    <div class="panel-heading" id="{{ category_name }}">
      <i class="far fa-folder"></i>&nbsp;
      </a>
    </div>
  </div>
</div>
  {% endif %}
{% endfor %}
{% endraw %}
{% endhighlight %}
{% endcapture %}{% include fixlinenos.html %}{{ _code }}

* Scrolling Horizontal

{% capture _code %}
{% highlight html linenos %}
{% raw %}
<div class="panel-group">
  <div class="panel panel-default">
    <div class="panel-heading" id="{{ category_name }}">
      <i class="far fa-folder"></i>
      <p>This is a very long long long long long long long long long long long long long long long long long long long long long line.</p>
      </a>
    </div>
  </div>
</div>
  {% endif %}
{% endfor %}
{% endraw %}
{% endhighlight %}
{% endcapture %}{% include fixlinenos.html %}{{ _code }}

#### MarkDown-html

```html
{% raw %}{% highlight html linenos %}{% endraw %}
{% raw %}{%{% endraw %} raw {% raw %}%}{% endraw %}
{% raw %}<p>{% if site.data.showAuthor %}Author:{{ site.data.author }}{% endif %}</p>
<p>This is some text in a paragraph.</p>{% endraw %}
{% raw %}{%{% endraw %} endraw {% raw %}%}{% endraw %}
{% raw %}{% endhighlight %}{% endraw %}
```

### Bash

{% capture _code %}
{% highlight bash linenos %}
server {

  listen   443 ssl;

  ssl on;
  server_name blog.cotes.in;

  ssl_certificate /etc/letsencrypt/live/blog.cotes.in/fullchain.pem;
  ssl_certificate_key /etc/letsencrypt/live/blog.cotes.in/privkey.pem;
}

{% endhighlight %}
{% endcapture %}{% include fixlinenos.html %}{{ _code }}

## Image Sample

![Desktop View](/assets/img/sample/mockup.jpg)


## Footnote Sample

Some long sentence. [^footnote]

[^footnote]: Test, [Link](https://google.com).
