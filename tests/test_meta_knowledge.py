"""Stage 4.9 meta-knowledge deduplication and promotion tests."""

from surrogate_rollout.optimization.meta_knowledge import MetaKnowledgeStore
from surrogate_rollout.optimization.schemas import MetaKnowledgeItem
from surrogate_rollout.prompt_routing.schemas import dumps_canonical


def knowledge(*, support, video, negative=(), scope="routing", confidence=0.8,
              status="candidate"):
    return MetaKnowledgeItem(
        knowledge_id=f"input_{support}", knowledge_type="routing_pattern",
        condition={"trait": "text_heavy"}, principle="select text entry",
        positive_support_ids=((support,) if support else ()),
        negative_support_ids=tuple(negative), distinct_video_ids=(video,),
        scope=scope, confidence=confidence, status=status,
        provenance={"source": support},
    )


def test_meta_knowledge_deduplicates_and_confirms_distinct_support():
    store = MetaKnowledgeStore()
    first = store.upsert(knowledge(support="e1", video="v1"))
    second = store.upsert(knowledge(support="e2", video="v2"))
    assert first.status == "candidate"
    assert second.status == "confirmed"
    assert second.positive_support_ids == ("e1", "e2")
    assert len(store.list_items()) == 1
    before = dumps_canonical(store.list_items())
    store.upsert(knowledge(support="e2", video="v2"))
    assert dumps_canonical(store.list_items()) == before


def test_contradictory_support_cannot_remain_confirmed():
    store = MetaKnowledgeStore()
    store.upsert(knowledge(support="e1", video="v1"))
    store.upsert(knowledge(support="e2", video="v2"))
    contradicted = store.upsert(knowledge(
        support="", video="v3", negative=("e3",), confidence=0.9))
    assert contradicted.status == "candidate"
    assert contradicted.confidence <= 0.5
    assert contradicted.provenance["contradictory_support"] is True


def test_single_example_never_promotes_global_scaffold_knowledge():
    store = MetaKnowledgeStore()
    single = store.upsert(knowledge(
        support="stage4_7_evidence", video="stage4_7_video",
        scope="global_scaffold", status="confirmed"))
    assert single.status == "candidate"
    second = store.upsert(knowledge(
        support="e2", video="v2", scope="global_scaffold"))
    assert second.status == "candidate"
    third = store.upsert(knowledge(
        support="e3", video="v3", scope="global_scaffold"))
    assert third.status == "confirmed"


def test_meta_knowledge_round_trip(tmp_path):
    path = tmp_path / "knowledge.json"
    store = MetaKnowledgeStore(str(path))
    store.upsert(knowledge(support="e1", video="v1"))
    store.save()
    restored = MetaKnowledgeStore(str(path))
    assert dumps_canonical(restored.list_items()) == \
        dumps_canonical(store.list_items())
