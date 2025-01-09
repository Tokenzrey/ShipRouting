import logging
from typing import Dict, Any

from fastapi import HTTPException

from models import DjikstraRequest, validate_djikstra_request  # Pastikan ini diadaptasi untuk FastAPI

# Konfigurasi Logging
logging.basicConfig(
    level=logging.DEBUG,  # Ubah ke DEBUG untuk informasi detail
    format='%(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

def djikstra_route_controller(grid_locator, route_optimizer, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Controller untuk menemukan rute menggunakan algoritma Dijkstra.
    Menyertakan partial paths dan semua edge dengan status blocking.
    """
    try:
        # Parse dan validasi request payload
        if not payload:
            raise ValueError("Invalid JSON payload.")
        
        # Validasi request menggunakan fungsi validate_djikstra_request
        validate_djikstra_request(payload)
        
        # Inisialisasi DjikstraRequest
        djikstra_request = DjikstraRequest(payload)  # Pastikan DjikstraRequest diinisialisasi dengan unpacking payload
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
        
        # Validasi node yang ditemukan
        if start_node < 0 or end_node < 0:
            logger.error("Node terdekat tidak valid.")
            raise HTTPException(status_code=400, detail="Invalid start or end coordinates.")
        
        # Temukan jalur terpendek menggunakan route_optimizer
        path, distance, partial_paths, all_edges = route_optimizer.find_shortest_path(
            start, end, use_model, ship_speed, condition
        )
        
        logger.debug(f"Path ditemukan: {path}")
        logger.debug(f"Total distance: {distance}")
        logger.debug(f"Jumlah partial paths: {len(partial_paths)}")
        logger.debug(f"Jumlah all_edges: {len(all_edges)}")
        
        # Jika tidak ada jalur ditemukan
        if not path:
            response = {
                "success": False,
                "error": "No path found between the given coordinates."
            }
            logger.warning("Tidak ada jalur yang ditemukan.")
            raise HTTPException(status_code=404, detail="No path found between the given coordinates.")
        
        # Berhasil, kembalikan jalur, jarak, partial_paths, dan all_edges
        response = {
            "success": True,
            "data": {
                "path": path,
                "distance": distance,
                "partial_paths": partial_paths,
                "all_edges": all_edges
            }
        }
        logger.info("Jalur ditemukan dan respons berhasil disiapkan.")
        return response
    
    except KeyError as e:
        logger.error(f"KeyError: {e}")
        raise HTTPException(status_code=400, detail=f"Missing required field: {e}")
    
    except ValueError as e:
        logger.error(f"ValueError: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except HTTPException as he:
        # Re-raise HTTPException untuk ditangani oleh FastAPI
        raise he
    
    except Exception as e:
        logger.error(f"Unexpected error dalam controller /djikstra: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")
