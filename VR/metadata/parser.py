from typing import Any, Dict, Optional, Tuple

Projection = str  # "equirect360" | "equirect180" | "fisheye180" | "unknown"
Layout = str      # "sbs" | "ou" | "mono" | "unknown"


def _get_tag(d: Dict[str, Any], key: str) -> Optional[str]:
    tags = d.get("tags") or {}
    if not isinstance(tags, dict):
        return None
    return tags.get(key)


def detect_from_ffprobe(meta: Dict[str, Any], width: int, height: int) -> Tuple[Projection, Layout, Dict[str, float]]:
    """
    Parse ffprobe JSON to infer projection, stereo layout, and initial pose (yaw/pitch/roll).
    Returns (projection, layout, pose_dict)
    """
    projection: Projection = "unknown"
    layout: Layout = "unknown"
    pose = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

    streams = meta.get("streams") or []
    # Search video stream(s)
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    for vs in video_streams:
        # Stereo 3D
        side_list = vs.get("side_data_list") or []
        for sd in side_list:
            sdt = sd.get("side_data_type") or sd.get("side_data_name")
            if isinstance(sdt, str) and "stereo" in sdt.lower():
                mode = sd.get("stereo_mode") or sd.get("type")
                if isinstance(mode, str):
                    m = mode.lower()
                    if "left_right" in m or "side by side" in m or "sbs" in m:
                        layout = "sbs"
                    elif "top_bottom" in m or "over_under" in m or "ou" in m:
                        layout = "ou"
                    elif "mono" in m:
                        layout = "mono"
            if isinstance(sdt, str) and ("spherical" in sdt.lower() or "projection" in sdt.lower()):
                proj = (sd.get("projection") or sd.get("type") or "").lower()
                if "equirect" in proj:
                    projection = "equirect360"  # default; refine below
                elif "cubemap" in proj:
                    projection = "cubemap"
                # yaw/pitch/roll
                for k in ("yaw", "pitch", "roll"):
                    if k in sd and isinstance(sd[k], (int, float)):
                        pose[k] = float(sd[k])
        # Some muxers put stereo in tags
        tag_sm = _get_tag(vs, "stereo_mode")
        if isinstance(tag_sm, str) and layout == "unknown":
            m = tag_sm.lower()
            if "left_right" in m or "side by side" in m or "sbs" in m:
                layout = "sbs"
            elif "top_bottom" in m or "over_under" in m or "ou" in m:
                layout = "ou"
            elif "mono" in m:
                layout = "mono"

    # GPano tags (often in format.tags or stream.tags)
    fmt = meta.get("format") or {}
    # Search both format.tags and video stream tags for GPano
    tag_sources = []
    if isinstance(fmt.get("tags"), dict):
        tag_sources.append(fmt["tags"])
    for vs in video_streams:
        if isinstance(vs.get("tags"), dict):
            tag_sources.append(vs["tags"])
    full_pano_width = None
    cropped = False
    for tags in tag_sources:
        proj_type = (tags.get("ProjectionType") or tags.get("GPano:ProjectionType") or "").lower()
        spherical = (tags.get("Spherical") or tags.get("GPano:Spherical") or "").lower()
        if "equirectangular" in proj_type or spherical == "true":
            if projection == "unknown":
                projection = "equirect360"
        # StereoMode in tags
        sm = (tags.get("StereoMode") or tags.get("GPano:StereoMode") or "")
        if isinstance(sm, str) and layout == "unknown":
            m = sm.lower()
            if "left-right" in m or "left_right" in m:
                layout = "sbs"
            elif "top-bottom" in m or "over-under" in m:
                layout = "ou"
            elif "mono" in m:
                layout = "mono"
        # 180 vs 360 via cropped area hints
        fpw = tags.get("FullPanoWidthPixels") or tags.get("GPano:FullPanoWidthPixels")
        caw = tags.get("CroppedAreaImageWidthPixels") or tags.get("GPano:CroppedAreaImageWidthPixels")
        if isinstance(fpw, str) and fpw.isdigit():
            full_pano_width = int(fpw)
        elif isinstance(fpw, (int, float)):
            full_pano_width = int(fpw)
        if isinstance(caw, str) and caw.isdigit():
            cropped = True
        elif isinstance(caw, (int, float)):
            cropped = True

    # Refine equirect 180 vs 360
    if projection.startswith("equirect"):
        # If FullPanoWidthPixels present and > video width, and CroppedArea hints exist -> 180
        if full_pano_width and full_pano_width > width and cropped:
            projection = "equirect180"
        else:
            # Heuristic: if content looks cropped to half pano (not available here), fallback to width:height≈2:1 -> 360
            projection = projection or "equirect360"

    # Fallback heuristics when unknown
    if projection == "unknown":
        aspect = width / float(height) if height else 0.0
        if abs(aspect - 2.0) < 0.15:
            projection = "equirect360"
        else:
            projection = "fisheye180"
    if layout == "unknown":
        if width % 2 == 0 and (width / float(height)) >= 1.6:
            layout = "sbs"
        elif height % 2 == 0 and (height / float(width)) >= 1.6:
            layout = "ou"
        else:
            layout = "mono"

    return projection, layout, pose

