---
layout: page
title:
  zh: 标签
  en: Tags
sitemap:
  priority: .5
---


{%comment%}
  'site.tags' looks like a Map, e.g. site.tags.MyTag.[ Post0, Post1, ... ]
  Print the {{ site.tags }} will help you to understand it.
{%endcomment%}
<div id="tags" class="d-flex flex-wrap">
{% assign tags = "" | split: "" %}
{% for t in site.tags %}
  {% assign tags = tags | push: t[0] %}
{% endfor %}

{% assign sorted_tags = tags | sort_natural %}

{% for t in sorted_tags %}
  <div>
    <a class="tag" href="/tags/{{ t | downcase | replace: ' ', '-' }}/">{{ t }}<span class="text-muted">{{ site.tags[t].size }}</span></a>
  </div>
{% endfor %}

</div>
