---
layout: page
permalink: /publications/
title: publications
description: publications in reversed chronological order.
nav: true
nav_order: 2
---

<!-- _pages/publications.md -->
<div class="publications">

{% bibliography %}

<h2 class="bibliography-heading mt-5">Under Review</h2>

{% bibliography --file under_review %}

</div>

<style>
.bibliography-heading {
  color: var(--global-text-color);
  font-size: 1.5rem;
  font-weight: 600;
  border-bottom: 2px solid var(--global-divider-color);
  padding-bottom: 0.5rem;
}
</style>
