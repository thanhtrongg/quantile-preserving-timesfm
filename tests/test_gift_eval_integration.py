import os
import unittest


@unittest.skipUnless(
    os.getenv("GIFT_EVAL") and os.getenv("RUN_GIFT_EVAL_INTEGRATION") == "1",
    "Set GIFT_EVAL and RUN_GIFT_EVAL_INTEGRATION=1 for the real-data integration test",
)
class GiftEvalIntegrationTest(unittest.TestCase):
    def test_official_dataset_path_is_available(self):
        from src.evaluation.gift_eval import load_gift_dataset

        dataset = load_gift_dataset("electricity/15T", term="short", to_univariate=True)
        self.assertGreater(dataset.prediction_length, 0)
        first = next(iter(dataset.test_data.input))
        self.assertIn("target", first)

