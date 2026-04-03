import json

from src.index_metadata import CONTEXTUALIZATION_VERSION
from src.text_splitter import TextSplitter


def test_split_report_adds_contextualized_fields_for_content_and_tables(tmp_path):
    captured_contexts = []

    def contextualizer(context):
        captured_contexts.append(context)
        return f"{context['chunk_type']} description"

    splitter = TextSplitter(contextualizer=contextualizer)
    report_data = {
        "metainfo": {"company_name": "Acme Corp"},
        "content": {
            "pages": [
                {
                    "page": 1,
                    "text": "# Management Discussion\nAnnual Report 2023\nRevenue increased meaningfully across segments."
                }
            ]
        },
    }
    serialized_report_path = tmp_path / "parsed_report.json"
    serialized_report_path.write_text(
        json.dumps(
            {
                "tables": [
                    {
                        "table_id": 7,
                        "page": 1,
                        "markdown": "Revenue by segment (USD in millions)\n| Metric | 2022 | 2023 |\n| --- | --- | --- |\n| Revenue | 10 | 12 |",
                        "serialized": {
                            "subject_core_entities_list": ["Revenue", "Operating income"],
                            "information_blocks": [
                                {"information_block": "Revenue was 12 in 2023."}
                            ],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    updated_report = splitter._split_report(report_data, serialized_report_path)

    assert updated_report["content"]["index_metadata"]["contextualization_version"] == CONTEXTUALIZATION_VERSION
    assert len(updated_report["content"]["chunks"]) == 2

    content_chunk = next(chunk for chunk in updated_report["content"]["chunks"] if chunk["type"] == "content")
    table_chunk = next(chunk for chunk in updated_report["content"]["chunks"] if chunk["type"] == "serialized_table")

    assert content_chunk["contextual_description"] == "content description"
    assert content_chunk["contextualized_text"].endswith(content_chunk["text"])
    assert content_chunk["nearest_heading"] == "Management Discussion"

    assert table_chunk["contextual_description"] == "serialized_table description"
    assert table_chunk["contextualized_text"].endswith(table_chunk["text"])
    assert table_chunk["table_topic"] == "Revenue by segment (USD in millions)"
    assert table_chunk["year_range"] == "2022-2023"
    assert table_chunk["unit"].lower() == "usd"
    assert table_chunk["primary_metrics"] == ["Revenue", "Operating income"]

    content_context = next(context for context in captured_contexts if context["chunk_type"] == "content")
    table_context = next(context for context in captured_contexts if context["chunk_type"] == "serialized_table")

    assert content_context["company_name"] == "Acme Corp"
    assert content_context["report_year"] == "2023"
    assert "Revenue increased" in content_context["page_text"]
    assert content_context["nearest_heading"] == "Management Discussion"

    assert table_context["table_topic"] == "Revenue by segment (USD in millions)"
    assert table_context["year_range"] == "2022-2023"
    assert table_context["unit"].lower() == "usd"
    assert table_context["primary_metrics"] == "Revenue, Operating income"


def test_split_all_reports_fails_fast_per_report(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    good_report = {
        "metainfo": {"company_name": "Good Co"},
        "content": {"pages": [{"page": 1, "text": "# Title\nAnnual Report 2023\nGood text."}]},
    }
    bad_report = {
        "metainfo": {"company_name": "Bad Co"},
        "content": {"pages": [{"page": 1, "text": "# Title\nAnnual Report 2023\nBad text."}]},
    }

    (input_dir / "good.json").write_text(json.dumps(good_report), encoding="utf-8")
    (input_dir / "bad.json").write_text(json.dumps(bad_report), encoding="utf-8")

    def contextualizer(context):
        if context["company_name"] == "Bad Co":
            raise RuntimeError("llm failed")
        return "ok"

    splitter = TextSplitter(contextualizer=contextualizer)
    splitter.split_all_reports(input_dir, output_dir)

    assert (output_dir / "good.json").exists()
    assert not (output_dir / "bad.json").exists()
