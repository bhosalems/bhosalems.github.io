---
layout: post
title: Quotodian commands
date: 2024-11-28 14:45:00 +0530
description:  Trace of all commands that use daily.
tags: note-to-self
categories: technical
pseudocode: true
math: true
---
##### Bash Commands

- Convert .vscode->launch.json debug to commandline
```Bash
echo ["vscode-text", "--args1", "--args2"] | tr -d '\n' | sed 's/[][]//g; s/"//g; s/,//g; s/[[:space:]]\+/ /g'
```