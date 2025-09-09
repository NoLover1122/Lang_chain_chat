from slowapi import Limiter
from slowapi.util import get_remote_address

# 创建限流器实例
limiter = Limiter(key_func=get_remote_address)

def get_scholar_open_id(request):
    """基于学者ID的限流键函数"""
    # 对于POST请求，尝试从请求体中获取学者ID
    try:
        # 注意：这只在已经解析的请求中可用
        if hasattr(request, 'state') and hasattr(request.state, 'scholar_open_id'):
            return f"scholar:{request.state.scholar_open_id}"
    except:
        pass
    
    # 回退到IP地址
    return get_remote_address(request) 