from src.config_loader.jd_config import load_jd_config


def test_load_jd_config(config):
    assert config.fit_mode == "rules"
    assert "embeddings" in config.expanded_must_have
    assert config.weights["title_career"] == 0.35


def test_config_has_disqualifier_titles(config):
    negs = config.role.get("negative_title_keywords", [])
    assert "marketing manager" in negs
    assert "hr manager" in negs
