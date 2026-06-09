import pandas as pd

from scripts.prepare_data import prepare_aime_dataset


def test_prepare_aime_dataset(tmp_path, monkeypatch):
    class DummyDataset(list):
        def shuffle(self, seed):
            return self

        def select(self, indices):
            if isinstance(indices, list):
                return [self[i] for i in indices]
            return self[:indices]

    dummy = DummyDataset([
        {"problem": "Compute 1+1", "answer": "2"},
        {"Question": "Compute 2+2", "Answer": "4"},
    ])

    monkeypatch.setattr('scripts.prepare_data.load_dataset', lambda *args, **kwargs: dummy)

    output_path = tmp_path / "train.csv"
    df = prepare_aime_dataset(output_path=str(output_path), count=2)

    assert output_path.exists()
    assert len(df) == 2
    assert list(df.columns) == ["id", "problem", "answer"]
    loaded = pd.read_csv(output_path)
    assert len(loaded) == 2
    assert str(loaded.iloc[0]["answer"]) in {"2", "4"}
