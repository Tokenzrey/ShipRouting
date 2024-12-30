from typing import Dict, Tuple

def validate_djikstra_request(data: Dict):
    """
    Validasi manual untuk request Dijkstra.
    """
    required_fields = ['start', 'end', 'ship_speed', 'condition']
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing field: {field}")

    for point in ['start', 'end']:
        if not isinstance(data[point], dict):
            raise ValueError(f"Field '{point}' harus berupa object dengan longitude dan latitude.")
        if 'longitude' not in data[point] or 'latitude' not in data[point]:
            raise ValueError(f"Missing longitude or latitude in {point}")

    if not isinstance(data['ship_speed'], (int, float)) or data['ship_speed'] <= 0:
        raise ValueError("Field 'ship_speed' harus berupa angka positif.")

    if not isinstance(data['condition'], int) or data['condition'] not in [0, 1]:
        raise ValueError("Field 'condition' harus berupa integer (0 atau 1).")

    if 'use_model' in data:
        if not isinstance(data['use_model'], bool):
            raise ValueError("Field 'use_model' harus berupa boolean.")

class DjikstraRequest:
    def __init__(self, data: Dict):
        """
        Inisialisasi dan validasi request Dijkstra.
        
        :param data: Dictionary yang berisi data request.
        :raises ValueError: Jika data request tidak valid.
        """
        validate_djikstra_request(data)
        
        self.start = (data['start']['longitude'], data['start']['latitude'])
        self.end = (data['end']['longitude'], data['end']['latitude'])
        self.ship_speed = data['ship_speed']
        self.condition = data['condition']
        self.use_model = data.get('use_model', False)
    
    def get_start(self) -> Tuple[float, float]:
        return self.start
    
    def get_end(self) -> Tuple[float, float]:
        return self.end

    def get_shipSpeed(self) -> float:
        return self.ship_speed

    def get_condition(self) -> int:
        return self.condition

    def get_use_model(self) -> bool:
        return self.use_model
