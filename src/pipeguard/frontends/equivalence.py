from __future__ import annotations

import json
from hashlib import sha256
from typing import Any

from ..model import Module


def _visibility(resource: Any) -> tuple[tuple[str, bool], ...]:
    mask = resource.visibility_mask
    return tuple(sorted(mask.artifact().items())) if mask is not None else ()


def semantic_signature(module: Module) -> dict[str, Any]:
    actor_order = sorted(
        module.actors,
        key=lambda actor: (
            actor.kind,
            actor.cta_rank if actor.cta_rank is not None else -1,
            actor.participants,
            actor.id,
        ),
    )
    actor_ids = {actor.id: f"actor:{index}" for index, actor in enumerate(actor_order)}

    allocation_regions: dict[tuple[str, int | None, str], list[tuple[int, int]]] = {}
    for region in module.regions:
        key = (region.address_space, region.owner_cta_rank, region.allocation_id)
        allocation_regions.setdefault(key, []).append((region.offset, region.extent))
    allocation_order = sorted(
        allocation_regions,
        key=lambda key: (
            key[0],
            key[1] if key[1] is not None else -1,
            tuple(sorted(allocation_regions[key])),
            key[2],
        ),
    )
    allocation_ids = {
        allocation: f"allocation:{index}"
        for index, allocation in enumerate(allocation_order)
    }
    region_order = sorted(
        module.regions,
        key=lambda region: (
            allocation_ids[(region.address_space, region.owner_cta_rank, region.allocation_id)],
            region.offset,
            region.extent,
            region.id,
        ),
    )
    region_ids = {region.id: f"region:{index}" for index, region in enumerate(region_order)}

    resource_order = sorted(
        module.resources,
        key=lambda resource: (
            resource.kind,
            resource.protocol or "",
            resource.scope or "",
            resource.capacity if resource.capacity is not None else -1,
            tuple(sorted(actor_ids[actor] for actor in resource.participants)),
            _visibility(resource),
            resource.id,
        ),
    )
    resource_ids = {
        resource.id: f"resource:{index}"
        for index, resource in enumerate(resource_order)
    }
    event_ids = {
        event.id: f"event:{actor_ids[event.actor]}:{event.seq:08d}"
        for event in module.events
    }

    actors = [
        {
            "id": actor_ids[actor.id],
            "kind": actor.kind,
            "participants": list(actor.participants),
            **({"cta_rank": actor.cta_rank} if actor.cta_rank is not None else {}),
        }
        for actor in actor_order
    ]
    allocations = [
        {
            "id": allocation_ids[key],
            "address_space": key[0],
            "owner_cta_rank": key[1],
            "intervals": [list(interval) for interval in sorted(allocation_regions[key])],
        }
        for key in allocation_order
    ]
    regions = [
        {
            "id": region_ids[region.id],
            "allocation_id": allocation_ids[(region.address_space, region.owner_cta_rank, region.allocation_id)],
            "address_space": region.address_space,
            "offset": region.offset,
            "extent": region.extent,
            **({"owner_cta_rank": region.owner_cta_rank} if region.owner_cta_rank is not None else {}),
        }
        for region in region_order
    ]
    resources = [
        {
            "id": resource_ids[resource.id],
            "kind": resource.kind,
            "participants": sorted(actor_ids[actor] for actor in resource.participants),
            **({"capacity": resource.capacity} if resource.capacity is not None else {}),
            **({"protocol": resource.protocol} if resource.protocol is not None else {}),
            **({"scope": resource.scope} if resource.scope is not None else {}),
            **(
                {"visibility_mask": resource.visibility_mask.artifact()}
                if resource.visibility_mask is not None
                else {}
            ),
        }
        for resource in resource_order
    ]

    events: list[dict[str, Any]] = []
    raw_events = {event["id"]: event for event in module.raw["events"]}
    for event in sorted(module.events, key=lambda item: (actor_ids[item.actor], item.seq, item.id)):
        raw = dict(raw_events[event.id])
        normalized: dict[str, Any] = {
            "id": event_ids[event.id],
            "type": event.type,
            "actor": actor_ids[event.actor],
            "seq": event.seq,
        }
        for key, value in raw.items():
            if key in {"id", "type", "actor", "seq", "parity_evidence"}:
                continue
            if key == "resource":
                normalized[key] = resource_ids[value]
            elif key in {"region", "source_region", "destination_region"}:
                normalized[key] = region_ids[value]
            elif key == "regions":
                normalized[key] = sorted(region_ids[item] for item in value)
            elif key in {"group_owner", "engine_id"}:
                normalized[key] = actor_ids[value]
            elif key == "depends_on":
                normalized[key] = sorted(event_ids[item] for item in value)
            else:
                normalized[key] = value
        if event.parity_evidence is not None:
            normalized["logical_wait_evidence"] = {
                "lifecycle_epoch": event.parity_evidence.lifecycle_epoch,
                "stream_rank": event.parity_evidence.stream_rank,
                "polling_coalesced": event.parity_evidence.polling_coalesced,
            }
        events.append(normalized)

    return {
        "schema_version": module.raw["schema_version"],
        "target": module.raw["target"],
        "assumptions": module.raw["assumptions"],
        "control_structure": module.raw["control_structure"],
        "actors": actors,
        "allocations": allocations,
        "regions": regions,
        "resources": resources,
        "events": events,
        "unsupported_constructs": module.raw["unsupported_constructs"],
    }


def semantic_digest(module: Module) -> str:
    encoded = json.dumps(
        semantic_signature(module),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()
