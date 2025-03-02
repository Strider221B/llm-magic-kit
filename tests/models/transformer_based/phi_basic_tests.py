import unittest
from unittest.mock import MagicMock, patch

from helpers.constants import Constants
from models.transformer_based.models import Phi

class TestPhi(unittest.TestCase):

    @patch('models.transformer_based.phi_basic.GPUHelper.clean_memory')
    def test_get_model_response_success(self, mock_clean_memory):
        # Mocking the tokenizer and model
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer.eos_token_id = 0
        mock_tokenizer.return_tensors = "pt"
        mock_tokenizer.decode.return_value = "Mocked response"
        
        prompt = "Test prompt"
        response = Phi.get_model_response(mock_model, mock_tokenizer, prompt)
        
        mock_tokenizer.assert_called_with(prompt, return_tensors="pt", padding=True)
        mock_model.generate.assert_called()
        mock_clean_memory.assert_called_once()
        self.assertEqual(response, "Mocked response")

    @patch('models.transformer_based.phi_basic.GPUHelper.clean_memory')
    def test_get_model_response_exception(self, mock_clean_memory):
        # Mocking the tokenizer and model
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer.eos_token_id = 0
        mock_tokenizer.return_tensors = "pt"
        mock_tokenizer.side_effect = Exception("Mocked exception")
        
        prompt = "Test prompt"
        response = Phi.get_model_response(mock_model, mock_tokenizer, prompt)
        
        self.assertEqual(response, f"Error occurred. Answer={Constants.DEFAULT_ANSWER}")
        mock_clean_memory.assert_not_called()

if __name__ == '__main__':
    unittest.main()