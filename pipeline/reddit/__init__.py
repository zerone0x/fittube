"""Extra Reddit sources on top of the comment search in ``fetch.py``.

Each module turns one kind of Reddit content into *evidence*: plain dicts in
the same shape as Arctic Shift comments (``id``, ``author``, ``body``,
``score``, ``created_utc``, ``subreddit``, ``link_id``) plus a ``kind`` key, so
``build.aggregate`` can treat every source the same way:

* ``wiki.py``  -- subreddit wiki pages (``kind="wiki"``, extra ``path``/``url``),
* ``posts.py`` -- submissions linking YouTube (``kind="post"``) and the
  "recommend me videos" threads whose replies are expanded like comments.

Modules expose ``collect(client, subreddits, ...) -> list[dict]`` and never
raise for a single failing subreddit: they log and move on.
"""
