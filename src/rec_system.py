import pandas as pd

def recommend(gdf, emotion):
    # Trả về các nhóm bài hát dựa theo cảm xúc
    if emotion == 'happy' or emotion == 'drowsy':
        return gdf.get_group(('high', 'high'))
    elif emotion == 'neutral':
        return gdf.get_group(('high', 'low'))
    elif emotion in ["angry", "disgust", "surprised"]:
        grp1 = gdf.get_group(('low', 'high'))
        grp2 = gdf.get_group(('high', 'low'))
        return pd.concat([grp1, grp2]) # Dùng concat thay vì append
    elif emotion in ["sad", "scared"]:
        grp1 = gdf.get_group(('low', 'low'))
        grp2 = gdf.get_group(('high', 'high'))
        return pd.concat([grp1, grp2])
    return None
