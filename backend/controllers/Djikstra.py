from flask import request, jsonify
import logging
from models import DjikstraRequest, validate_djikstra_request

logger = logging.getLogger(__name__)

def djikstra_route(grid_locator, route_optimizer):
    try:
        # Parse dan validasi request JSON
        data = request.get_json()
        logger.debug(f"Menerima request dengan parameter: {request.args}")
        logger.debug(f"Menerima request body: {request.get_json()}")
        if not data:
            raise ValueError("Invalid JSON payload.")
        
        # Validasi request menggunakan fungsi validate_djikstra_request
        validate_djikstra_request(data)
        
        # Inisialisasi DjikstraRequest
        djikstra_request = DjikstraRequest(data)
        start = djikstra_request.get_start()
        end = djikstra_request.get_end()
        ship_speed = djikstra_request.get_shipSpeed()
        condition = djikstra_request.get_condition()
        use_model = djikstra_request.get_use_model()
        
        logger.info(f"Request diterima: Start={start}, End={end}, Use Model={use_model}")
        
        # Gunakan grid_locator untuk menemukan node terdekat
        start_node = grid_locator.find_nearest_node(*start)
        end_node = grid_locator.find_nearest_node(*end)
        logger.info(f"Start node: {start_node}, End node: {end_node}")
        
        # Temukan jalur terpendek menggunakan route_optimizer
        path, distance = route_optimizer.find_shortest_path(start, end, use_model, ship_speed, condition)
        route = [list(map(float, node.split('_'))) for node in path]
        # Jika tidak ada jalur ditemukan
        if not path:
            response = {"success": False, "error": "No path found between the given coordinates."}
            return jsonify(response), 404
        
        # Berhasil, kembalikan jalur dan jarak
        response = {
            "success": True,
            "data": {
                "path": route,
                "distance": distance
            }
        }
        return jsonify(response), 200
    
    except KeyError as e:
        logger.error(f"KeyError: {e}")
        response = {"success": False, "error": f"Missing required field: {e}"}
        return jsonify(response), 400

    except ValueError as e:
        logger.error(f"ValueError: {e}")
        response = {"success": False, "error": str(e)}
        return jsonify(response), 400

    except Exception as e:
        logger.error(f"Unexpected error dalam controller /djikstra: {e}")
        response = {"success": False, "error": "Internal server error."}
        return jsonify(response), 500
