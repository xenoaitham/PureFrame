"""Plugin API for custom detectors.

A detector plugin is a class honoring the same shape as
``NudityDetector`` (see ``docs/plugin-api.md`` for the full contract):

    class MyDetector:
        label_categories = {"pistol": "WEAPON_VISIBLE"}
        label_thresholds = {"pistol": 0.6}

        def __init__(self, settings: ProfileSettings): ...
        def detect_batch(self, frames_bgr) -> list[list[Detection]]: ...
        def unload(self) -> None: ...

Plugins register through the ``pureframe.plugins`` entry-point group, so
installation is the only step:

    [project.entry-points."pureframe.plugins"]
    weapons = "pureframe_weapons:MyDetector"

``discover()`` resolves the group into registrations without constructing
any detector: construction is expensive (model weights) and the pipeline
builds plugins only when they are enabled.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from importlib.metadata import EntryPoint, entry_points

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "pureframe.plugins"

# Threshold used for labels a plugin ships without an explicit value in
# ``label_thresholds``. Deliberately below the nudity default (0.55): a
# plugin author maps narrow labels and a missed detection is worse than a
# spurious box for this tool's purpose.
DEFAULT_LABEL_THRESHOLD = 0.5


@dataclass(frozen=True)
class PluginRegistration:
    """One discovered plugin: its class plus its label mapping.

    ``label_categories`` maps raw detector labels to category names -
    either existing ``Category`` values (``NUDITY_EXPLICIT``, ...) or new
    box-provider names the plugin introduces (``WEAPON_VISIBLE``, ...).
    ``label_thresholds`` carries per-label default thresholds; labels
    without one fall back to :data:`DEFAULT_LABEL_THRESHOLD`.
    """

    name: str
    cls: type
    label_categories: dict[str, str] = field(default_factory=dict)
    label_thresholds: dict[str, float] = field(default_factory=dict)

    def threshold_for(self, label: str) -> float:
        value = self.label_thresholds.get(label, DEFAULT_LABEL_THRESHOLD)
        return max(0.0, min(float(value), 1.0))

    def category_names(self) -> set[str]:
        return set(self.label_categories.values())


def validate_plugin(cls: type) -> list[str]:
    """Return the contract violations of ``cls`` (empty list = valid).

    Checks what the pipeline actually calls: construction with a single
    ``ProfileSettings``-style argument, ``detect_batch`` on a frame list,
    and ``unload``. The label maps are optional at the class level (a
    plugin without them registers but can never flag anything).
    """
    problems: list[str] = []
    if not isinstance(cls, type):
        problems.append(f"entry point resolved to {cls!r}, not a class")
        return problems
    try:
        try:
            signature = inspect.signature(cls)
        except (TypeError, ValueError):
            signature = None
        if signature is not None:
            params = [
                p
                for name, p in signature.parameters.items()
                if name not in ("self",)
                and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
            ]
            if len(params) > 1:
                problems.append(
                    "__init__ must take a single settings argument "
                    f"(found {len(params)} required parameters)"
                )
    except Exception:  # pragma: no cover - exotic metaclasses only
        pass
    if not callable(getattr(cls, "detect_batch", None)):
        problems.append("missing a callable detect_batch(frames_bgr) method")
    if not callable(getattr(cls, "unload", None)):
        problems.append("missing an unload() method")
    return problems


def _resolve(entry_point: EntryPoint) -> PluginRegistration:
    cls = entry_point.load()
    problems = validate_plugin(cls)
    if problems:
        raise TypeError(
            f"plugin {entry_point.name!r} breaks the contract: {'; '.join(problems)}"
        )
    label_categories = getattr(cls, "label_categories", None) or {}
    label_thresholds = getattr(cls, "label_thresholds", None) or {}
    if not isinstance(label_categories, dict) or not isinstance(label_thresholds, dict):
        raise TypeError(
            f"plugin {entry_point.name!r}: label_categories and label_thresholds "
            "must be dicts"
        )
    for label, category in label_categories.items():
        if not isinstance(label, str) or not isinstance(category, str) or not category:
            raise TypeError(
                f"plugin {entry_point.name!r}: label_categories must map "
                f"non-empty strings to non-empty strings (got {label!r} -> {category!r})"
            )
    cleaned_thresholds: dict[str, float] = {}
    for label, threshold in label_thresholds.items():
        if not isinstance(label, str):
            raise TypeError(
                f"plugin {entry_point.name!r}: label_thresholds keys must be "
                f"strings (got {label!r})"
            )
        if isinstance(threshold, bool) or not isinstance(threshold, int | float):
            raise TypeError(
                f"plugin {entry_point.name!r}: threshold for {label!r} must be "
                f"a number (got {threshold!r})"
            )
        cleaned_thresholds[label] = float(threshold)
    return PluginRegistration(
        name=entry_point.name,
        cls=cls,
        label_categories=dict(label_categories),
        label_thresholds=cleaned_thresholds,
    )


def discover() -> dict[str, PluginRegistration]:
    """Resolve the ``pureframe.plugins`` entry-point group.

    Returns ``{name: PluginRegistration}``. Plugins that fail to import or
    break the contract are logged and skipped - one broken third-party
    package must not take the CLI down; ``plugins list`` surfaces what was
    skipped through the log. No detector is constructed here.
    """
    registry: dict[str, PluginRegistration] = {}
    try:
        found = entry_points(group=ENTRY_POINT_GROUP)
    except TypeError:  # pragma: no cover - Python < 3.10 selectable API
        found = [ep for ep in entry_points() if ep.group == ENTRY_POINT_GROUP]
    for ep in found:
        try:
            registry[ep.name] = _resolve(ep)
        except Exception as e:
            logger.warning("Skipping PureFrame plugin %r: %s", ep.name, e)
    return registry
