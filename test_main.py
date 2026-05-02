import pytest
import os
from unittest.mock import patch, MagicMock
import pandas as pd

from main import LABEL_MAP, main


class TestMain:
    """Test suite for main.py functions."""

    def test_label_map_is_dict(self):
        """Test that LABEL_MAP is a dictionary."""
        assert isinstance(LABEL_MAP, dict)
        assert len(LABEL_MAP) > 0

    def test_label_map_keys_are_strings(self):
        """Test that all keys in LABEL_MAP are strings."""
        for key in LABEL_MAP.keys():
            assert isinstance(key, str)

    def test_label_map_values_are_strings(self):
        """Test that all values in LABEL_MAP are strings."""
        for value in LABEL_MAP.values():
            assert isinstance(value, str)

    def test_label_map_id_mapping(self):
        """Test id column mapping."""
        assert 'id' in LABEL_MAP
        assert LABEL_MAP['id'] == 'id'

    def test_label_map_review_mapping(self):
        """Test review column mapping."""
        assert 'review' in LABEL_MAP
        assert LABEL_MAP['review'] == '评论'

    def test_label_map_star_mapping(self):
        """Test star column mapping."""
        assert 'star' in LABEL_MAP
        assert LABEL_MAP['star'] == '评级'

    def test_label_map_location_mappings(self):
        """Test location-related column mappings."""
        assert 'Location#Transportation' in LABEL_MAP
        assert LABEL_MAP['Location#Transportation'] == '交通方便'
        assert 'Location#Downtown' in LABEL_MAP
        assert LABEL_MAP['Location#Downtown'] == '位于商圈附近'
        assert 'Location#Easy_to_find' in LABEL_MAP
        assert LABEL_MAP['Location#Easy_to_find'] == '是否容易寻找'

    def test_label_map_service_mappings(self):
        """Test service-related column mappings."""
        assert 'Service#Queue' in LABEL_MAP
        assert LABEL_MAP['Service#Queue'] == '排队时间'
        assert 'Service#Hospitality' in LABEL_MAP
        assert LABEL_MAP['Service#Hospitality'] == '服务人员态度'
        assert 'Service#Parking' in LABEL_MAP
        assert LABEL_MAP['Service#Parking'] == '停车方便'
        assert 'Service#Timely' in LABEL_MAP
        assert LABEL_MAP['Service#Timely'] == '点菜/上菜速度'

    def test_label_map_price_mappings(self):
        """Test price-related column mappings."""
        assert 'Price#Level' in LABEL_MAP
        assert LABEL_MAP['Price#Level'] == '价格'
        assert 'Price#Cost_effective' in LABEL_MAP
        assert LABEL_MAP['Price#Cost_effective'] == '性价比'
        assert 'Price#Discount' in LABEL_MAP
        assert LABEL_MAP['Price#Discount'] == '折扣力度'

    def test_label_map_ambience_mappings(self):
        """Test ambience-related column mappings."""
        assert 'Ambience#Decoration' in LABEL_MAP
        assert LABEL_MAP['Ambience#Decoration'] == '装修'
        assert 'Ambience#Noise' in LABEL_MAP
        assert LABEL_MAP['Ambience#Noise'] == '嘈杂情况'
        assert 'Ambience#Space' in LABEL_MAP
        assert LABEL_MAP['Ambience#Space'] == '就餐空间'
        assert 'Ambience#Sanitary' in LABEL_MAP
        assert LABEL_MAP['Ambience#Sanitary'] == '卫生情况'

    def test_label_map_food_mappings(self):
        """Test food-related column mappings."""
        assert 'Food#Portion' in LABEL_MAP
        assert LABEL_MAP['Food#Portion'] == '分量'
        assert 'Food#Taste' in LABEL_MAP
        assert LABEL_MAP['Food#Taste'] == '口味'
        assert 'Food#Appearance' in LABEL_MAP
        assert LABEL_MAP['Food#Appearance'] == '外观'
        assert 'Food#Recommend' in LABEL_MAP
        assert LABEL_MAP['Food#Recommend'] == '食物推荐程度'

    def test_label_map_complete_mapping_count(self):
        """Test that LABEL_MAP has expected number of mappings."""
        expected_keys = [
            'id', 'review', 'star',
            'Location#Transportation', 'Location#Downtown', 'Location#Easy_to_find',
            'Service#Queue', 'Service#Hospitality', 'Service#Parking', 'Service#Timely',
            'Price#Level', 'Price#Cost_effective', 'Price#Discount',
            'Ambience#Decoration', 'Ambience#Noise', 'Ambience#Space', 'Ambience#Sanitary',
            'Food#Portion', 'Food#Taste', 'Food#Appearance', 'Food#Recommend'
        ]
        assert len(LABEL_MAP) == len(expected_keys)
        for key in expected_keys:
            assert key in LABEL_MAP

    def test_label_map_no_empty_values(self):
        """Test that LABEL_MAP has no empty string values."""
        for value in LABEL_MAP.values():
            assert value != ""

    def test_label_map_no_empty_keys(self):
        """Test that LABEL_MAP has no empty string keys."""
        for key in LABEL_MAP.keys():
            assert key != ""

    @patch('main.load_dataset')
    def test_main_loads_dataset(self, mock_load_dataset):
        """Test that main() loads dataset from correct path."""
        mock_dataset = MagicMock()
        mock_dataset.rename_columns.return_value = mock_dataset
        mock_dataset.remove_columns.return_value = mock_dataset
        mock_dataset.select.return_value = mock_dataset
        mock_dataset.map.return_value = mock_dataset
        mock_load_dataset.return_value = mock_dataset

        main()

        mock_load_dataset.assert_called_once()
        call_args = mock_load_dataset.call_args
        assert 'csv' in call_args[1]['data_files']

    @patch('main.load_dataset')
    @patch('builtins.print')
    def test_main_renames_columns(self, mock_print, mock_load_dataset):
        """Test that main() renames columns using LABEL_MAP."""
        mock_dataset = MagicMock()
        mock_dataset.rename_columns.return_value = mock_dataset
        mock_dataset.remove_columns.return_value = mock_dataset
        mock_dataset.select.return_value = mock_dataset
        mock_dataset.map.return_value = mock_dataset
        mock_load_dataset.return_value = mock_dataset

        main()

        mock_dataset.rename_columns.assert_called_once_with(LABEL_MAP)

    @patch('main.load_dataset')
    @patch('builtins.print')
    def test_main_removes_id_and_rating(self, mock_print, mock_load_dataset):
        """Test that main() removes id and rating columns."""
        mock_dataset = MagicMock()
        mock_dataset.rename_columns.return_value = mock_dataset
        mock_dataset.remove_columns.return_value = mock_dataset
        mock_dataset.select.return_value = mock_dataset
        mock_dataset.map.return_value = mock_dataset
        mock_load_dataset.return_value = mock_dataset

        main()

        mock_dataset.remove_columns.assert_called_once()
        removed_cols = mock_dataset.remove_columns.call_args[0][0]
        assert 'id' in removed_cols
        assert '评级' in removed_cols

    @patch('main.load_dataset')
    @patch('builtins.print')
    def test_main_selects_first_10_samples(self, mock_print, mock_load_dataset):
        """Test that main() selects first 10 samples."""
        mock_dataset = MagicMock()
        mock_dataset.rename_columns.return_value = mock_dataset
        mock_dataset.remove_columns.return_value = mock_dataset
        mock_dataset.select.return_value = mock_dataset
        mock_dataset.map.return_value = mock_dataset
        mock_load_dataset.return_value = mock_dataset

        main()

        mock_dataset.select.assert_called()
        select_args = mock_dataset.select.call_args[0][0]
        assert list(select_args) == list(range(10))

    @patch('main.load_dataset')
    @patch('builtins.print')
    @patch('main.add_tsa_columns')
    def test_main_maps_add_tsa_columns(self, mock_add_tsa, mock_print, mock_load_dataset):
        """Test that main() applies add_tsa_columns mapping."""
        mock_dataset = MagicMock()
        mock_dataset.rename_columns.return_value = mock_dataset
        mock_dataset.remove_columns.return_value = mock_dataset
        mock_dataset.select.return_value = mock_dataset
        mock_dataset.map.return_value = mock_dataset
        mock_load_dataset.return_value = mock_dataset

        main()

        mock_dataset.map.assert_called()
        call_kwargs = mock_dataset.map.call_args[1]
        assert call_kwargs['load_from_cache_file'] is False

    @patch('main.load_dataset')
    @patch('builtins.print')
    def test_main_prints_result(self, mock_print, mock_load_dataset):
        """Test that main() prints the processed sample."""
        mock_dataset = MagicMock()
        mock_dataset.rename_columns.return_value = mock_dataset
        mock_dataset.remove_columns.return_value = mock_dataset
        mock_dataset.select.return_value = mock_dataset
        mock_dataset.map.return_value = mock_dataset
        mock_load_dataset.return_value = mock_dataset

        main()

        mock_print.assert_called()

    @patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'test_key'})
    @patch('main.load_dataset')
    def test_main_handles_missing_data_dir(self, mock_load_dataset):
        """Test main behavior when data directory might be missing."""
        with patch('os.path.join') as mock_join:
            mock_join.return_value = "./data/train.csv"
            # Should handle gracefully
            mock_load_dataset.side_effect = Exception("File not found")
            with pytest.raises(Exception):
                main()
