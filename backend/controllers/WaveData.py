import logging
from typing import Dict, Any, Optional

from fastapi import HTTPException

from managers import get_wave_data_response_interpolate

# Konfigurasi Logging
logging.basicConfig(
    level=logging.DEBUG,  # Ubah ke DEBUG untuk informasi detail
    format='%(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

def get_wave_data_controller(
    route_optimizer, date: Optional[str], time_slot: Optional[str], currentdate: bool
) -> Dict[str, Any]:
    """
    Controller untuk mendapatkan data gelombang.
    """
    try:
        response, status_code = get_wave_data_response_interpolate(
            {},  # Masukkan parameter tambahan jika diperlukan
            route_optimizer,
            date_str=date,
            time_slot=time_slot,
            currentdate=currentdate
        )

        if not status_code == 200:
            raise ValueError("error", "Unknown error")

        # Langsung kembalikan `data` dari response
        return {'data': response, 'success': True}
    
    except ValueError as ve:
        logger.warning(f"Kesalahan validasi: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    
    except Exception as e:
        logger.error(f"Error tidak terduga dalam controller get_wave_data: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")
