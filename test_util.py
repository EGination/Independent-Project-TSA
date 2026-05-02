import pytest
from unittest.mock import patch, MagicMock

from util import add_tsa_columns, TEST_JSON, api


class TestUtil:
    """Test suite for util.py functions."""

    def test_test_json_structure(self):
        """Test that TEST_JSON has expected structure."""
        assert "目标列表" in TEST_JSON
        assert isinstance(TEST_JSON["目标列表"], list)
        assert len(TEST_JSON["目标列表"]) == 1
        assert "目标" in TEST_JSON["目标列表"][0]
        assert "情感" in TEST_JSON["目标列表"][0]
        assert "理由" in TEST_JSON["目标列表"][0]

    def test_test_json_emotion_value(self):
        """Test that TEST_JSON has valid emotion value."""
        assert TEST_JSON["目标列表"][0]["情感"] == 1

    @patch('util.DeepSeekAPI')
    def test_add_tsa_columns_basic(self, mock_api_class):
        """Test add_tsa_columns with basic input."""
        mock_response = MagicMock()
        mock_response.get.return_value = TEST_JSON["目标列表"]
        mock_api_instance = MagicMock()
        mock_api_instance.generate_tsa.return_value = mock_response
        mock_api_class.return_value = mock_api_instance

        example = {
            "评论": "这家餐厅很好吃"
        }
        result = add_tsa_columns(example)

        assert "目标" in result
        assert "标签" in result
        assert "理由" in result
        assert "目标数量" in result
        assert result["评论"] == "这家餐厅很好吃"

    @patch('util.DeepSeekAPI')
    def test_add_tsa_columns_extracts_targets(self, mock_api_class):
        """Test that targets are correctly extracted from API response."""
        target_list = [
            {"目标": "餐厅环境", "情感": 1, "理由": "环境很好"},
            {"目标": "服务质量", "情感": -1, "理由": "服务一般"}
        ]
        mock_response = MagicMock()
        mock_response.get.return_value = target_list
        mock_api_instance = MagicMock()
        mock_api_instance.generate_tsa.return_value = mock_response
        mock_api_class.return_value = mock_api_instance

        example = {"评论": "测试评论"}
        result = add_tsa_columns(example)

        assert result["目标"] == ["餐厅环境", "服务质量"]
        assert result["标签"] == [1, -1]
        assert result["理由"] == ["环境很好", "服务一般"]
        assert result["目标数量"] == 2

    @patch('util.DeepSeekAPI')
    def test_add_tsa_columns_empty_target_list(self, mock_api_class):
        """Test add_tsa_columns when API returns empty target list."""
        mock_response = MagicMock()
        mock_response.get.return_value = []
        mock_api_instance = MagicMock()
        mock_api_instance.generate_tsa.return_value = mock_response
        mock_api_class.return_value = mock_api_instance

        example = {"评论": "无目标评论"}
        result = add_tsa_columns(example)

        assert result["目标"] == []
        assert result["标签"] == []
        assert result["理由"] == []
        assert result["目标数量"] == 0

    @patch('util.DeepSeekAPI')
    def test_add_tsa_columns_missing_key(self, mock_api_class):
        """Test add_tsa_columns when API response is missing '目标列表' key."""
        mock_response = MagicMock()
        mock_response.get.return_value = None
        mock_api_instance = MagicMock()
        mock_api_instance.generate_tsa.return_value = mock_response
        mock_api_class.return_value = mock_api_instance

        example = {"评论": "测试"}
        result = add_tsa_columns(example)

        assert result["目标"] == []
        assert result["标签"] == []
        assert result["理由"] == []
        assert result["目标数量"] == 0

    @patch('util.DeepSeekAPI')
    def test_add_tsa_columns_no_targets_key(self, mock_api_class):
        """Test add_tsa_columns when API response has no '目标列表' key."""
        mock_response = {"其他键": []}
        mock_api_instance = MagicMock()
        mock_api_instance.generate_tsa.return_value = mock_response
        mock_api_class.return_value = mock_api_instance

        example = {"评论": "测试"}
        result = add_tsa_columns(example)

        assert result["目标"] == []
        assert result["标签"] == []
        assert result["理由"] == []
        assert result["目标数量"] == 0

    @patch('util.DeepSeekAPI')
    def test_add_tsa_columns_negative_emotion(self, mock_api_class):
        """Test add_tsa_columns with negative emotion values."""
        target_list = [
            {"目标": "价格", "情感": -2, "理由": "价格太贵"}
        ]
        mock_response = MagicMock()
        mock_response.get.return_value = target_list
        mock_api_instance = MagicMock()
        mock_api_instance.generate_tsa.return_value = mock_response
        mock_api_class.return_value = mock_api_instance

        example = {"评论": "价格贵"}
        result = add_tsa_columns(example)

        assert result["标签"] == [-2]

    @patch('util.DeepSeekAPI')
    def test_add_tsa_columns_single_target(self, mock_api_class):
        """Test add_tsa_columns with single target."""
        target_list = [
            {"目标": "口味", "情感": 2, "理由": "非常好吃"}
        ]
        mock_response = MagicMock()
        mock_response.get.return_value = target_list
        mock_api_instance = MagicMock()
        mock_api_instance.generate_tsa.return_value = mock_response
        mock_api_class.return_value = mock_api_instance

        example = {"评论": "口味好"}
        result = add_tsa_columns(example)

        assert len(result["目标"]) == 1
        assert result["目标数量"] == 1

    @patch('util.DeepSeekAPI')
    def test_add_tsa_columns_multiple_targets(self, mock_api_class):
        """Test add_tsa_columns with multiple targets (edge case)."""
        target_list = [
            {"目标": f"方面{i}", "情感": 1, "理由": f"理由{i}"}
            for i in range(10)
        ]
        mock_response = MagicMock()
        mock_response.get.return_value = target_list
        mock_api_instance = MagicMock()
        mock_api_instance.generate_tsa.return_value = mock_response
        mock_api_class.return_value = mock_api_instance

        example = {"评论": "多方面评价"}
        result = add_tsa_columns(example)

        assert len(result["目标"]) == 10
        assert result["目标数量"] == 10

    @patch('util.DeepSeekAPI')
    def test_add_tsa_columns_preserves_original_comment(self, mock_api_class):
        """Test that original comment is preserved."""
        mock_response = MagicMock()
        mock_response.get.return_value = []
        mock_api_instance = MagicMock()
        mock_api_instance.generate_tsa.return_value = mock_response
        mock_api_class.return_value = mock_api_instance

        original_comment = "这是一条很长的评论" * 10
        example = {"评论": original_comment}
        result = add_tsa_columns(example)

        assert result["评论"] == original_comment

    @patch('util.DeepSeekAPI')
    def test_add_tsa_columns_missing_emotion_key(self, mock_api_class):
        """Test add_tsa_columns when target is missing '情感' key."""
        target_list = [
            {"目标": "测试目标", "理由": "理由"}
        ]
        mock_response = MagicMock()
        mock_response.get.return_value = target_list
        mock_api_instance = MagicMock()
        mock_api_instance.generate_tsa.return_value = mock_response
        mock_api_class.return_value = mock_api_instance

        example = {"评论": "测试"}
        # Should raise KeyError - this is expected behavior
        with pytest.raises(KeyError):
            add_tsa_columns(example)

    @patch('util.DeepSeekAPI')
    def test_add_tsa_columns_missing_target_key(self, mock_api_class):
        """Test add_tsa_columns when target is missing '目标' key."""
        target_list = [
            {"情感": 1, "理由": "理由"}
        ]
        mock_response = MagicMock()
        mock_response.get.return_value = target_list
        mock_api_instance = MagicMock()
        mock_api_instance.generate_tsa.return_value = mock_response
        mock_api_class.return_value = mock_api_instance

        example = {"评论": "测试"}
        with pytest.raises(KeyError):
            add_tsa_columns(example)

    @patch('util.DeepSeekAPI')
    def test_add_tsa_columns_with_chinese_comment(self, mock_api_class):
        """Test add_tsa_columns with Chinese text."""
        target_list = [
            {"目标": "环境", "情感": 1, "理由": "装修豪华"}
        ]
        mock_response = MagicMock()
        mock_response.get.return_value = target_list
        mock_api_instance = MagicMock()
        mock_api_instance.generate_tsa.return_value = mock_response
        mock_api_class.return_value = mock_api_instance

        example = {"评论": "这家餐厅装修非常豪华，服务也很好"}
        result = add_tsa_columns(example)

        assert "环境" in result["目标"]
