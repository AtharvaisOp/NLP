import unittest

from ml.preprocessing import preprocess_text


class PreprocessingContractTests(unittest.TestCase):
    def test_model_path_preserves_context_and_code_mixing(self) -> None:
        prepared = preprocess_text("हे product चांगले नाही!!! #review @user")

        self.assertIn("product", prepared.model_text)
        self.assertIn("नाही", prepared.model_text)
        self.assertIn("!!!", prepared.model_text)
        self.assertIn("#review", prepared.model_text)
        self.assertIn("<USER>", prepared.model_text)

    def test_model_path_is_nfc_normalized(self) -> None:
        prepared = preprocess_text("Cafe\u0301")

        self.assertEqual(prepared.model_text, "Caf\u00e9")

    def test_analysis_path_tokenizes_without_shared_stopword_or_lemma_pass(self) -> None:
        prepared = preprocess_text("हे products चांगले नाही!!!")

        self.assertEqual(prepared.analysis_text, "हे products चांगले नाही")
        self.assertEqual(prepared.analysis_tokens, ("हे", "products", "चांगले", "नाही"))


if __name__ == "__main__":
    unittest.main()
