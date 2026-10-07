"""Regenerate path_diagram.html (the published artifact page) from max_path.svg, the source of truth.

    python3 build_diagram.py

max_path.svg carries its own <style> (theme tokens and the diagram's classes). The page reuses that
CSS verbatim and inlines the <svg>, so the picture in the docs and on the page cannot drift apart.
Page layout lives in path_diagram.template.html.
"""
import os
import re

here = os.path.dirname(os.path.abspath(__file__))
svg = open(os.path.join(here, "max_path.svg")).read()
css = re.search(r"<style>(.*?)</style>", svg, re.S).group(1)
inline = re.sub(r"<\?xml.*?\?>\s*|<!--.*?-->\s*", "", svg, count=2, flags=re.S)
inline = re.sub(r"\s*<style>.*?</style>", "", inline, count=1, flags=re.S)
page = open(os.path.join(here, "path_diagram.template.html")).read()
page = page.replace("{{CSS}}", css.strip("\n")).replace("{{SVG}}", inline.strip())
open(os.path.join(here, "path_diagram.html"), "w").write(page)
print("wrote path_diagram.html", len(page), "bytes")
