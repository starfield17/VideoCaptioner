"""Small runtime Qt translation catalog for English and Simplified Chinese."""

from PySide6.QtCore import QCoreApplication, QTranslator

_ZH = {
    "VideoCaptioner": "视频字幕工作台",
    "Caption production console": "字幕生产控制台",
    "Run": "运行",
    "Settings": "设置",
    "Models": "模型",
    "Diagnostics": "诊断",
    "Ready": "就绪",
    "Running": "运行中",
    "Cancelling": "正在取消",
    "Completed": "已完成",
    "Failed": "失败",
    "Mode": "模式",
    "Full pipeline": "完整流程",
    "Transcribe only": "仅转写",
    "Refine subtitles": "优化已有字幕",
    "Input": "输入",
    "Output folder": "输出目录",
    "Browse": "浏览",
    "Browse file": "选择文件",
    "Browse folder": "选择目录",
    "ASR profile": "ASR 配置",
    "Source language": "源语言",
    "Target language": "目标语言",
    "Correction": "校正",
    "Translation": "翻译",
    "Repair": "修复",
    "Bilingual": "双语",
    "Bilingual input": "双语输入",
    "Start": "开始",
    "Cancel": "取消",
    "Stage": "阶段",
    "Activity": "运行记录",
    "Results": "结果",
    "File": "文件",
    "Status": "状态",
    "Outputs": "输出",
    "Open": "打开",
    "Configuration": "配置",
    "LLM provider": "LLM 服务",
    "Base URL": "Base URL",
    "Model ID": "模型 ID",
    "API key": "API 密钥",
    "Structured output": "结构化输出",
    "Timeout (seconds)": "超时（秒）",
    "Attempts": "重试次数",
    "Batch size": "批大小",
    "Workers": "并发数",
    "Model endpoint": "模型源",
    "Offline mode": "离线模式",
    "Log level": "日志级别",
    "Save settings": "保存设置",
    "Settings saved": "设置已保存",
    "The API key is stored as plaintext in the local config file.": (
        "API 密钥将以明文写入本地配置文件。"
    ),
    "Model directory": "模型目录",
    "Refresh": "刷新",
    "Download selected": "下载所选模型",
    "Key": "标识",
    "Provider": "服务",
    "Installed": "已安装",
    "Location": "位置",
    "Notes": "说明",
    "Yes": "是",
    "No": "否",
    "Run doctor": "运行诊断",
    "Load model": "加载模型",
    "Check": "检查项",
    "Detail": "详情",
    "Open log folder": "打开日志目录",
    "Select input": "选择输入",
    "Select output folder": "选择输出目录",
    "Select a model first.": "请先选择模型。",
    "Operation failed": "操作失败",
    "Invalid settings": "设置无效",
    "A task is still running. Request cancellation and close afterwards?": (
        "任务仍在运行。是否请求取消并在结束后关闭？"
    ),
    "Confirm close": "确认关闭",
    "Language": "语言",
    "English": "English",
    "Simplified Chinese": "简体中文",
    "No results yet": "暂无结果",
    "Download complete": "下载完成",
    "Doctor complete": "诊断完成",
}


class CatalogTranslator(QTranslator):
    """Translate stable English source strings without generated resources."""

    def __init__(self, language: str) -> None:
        super().__init__()
        self._language = language

    def translate(
        self,
        context: str,
        source_text: str,
        disambiguation: str | None = None,
        n: int = -1,
    ) -> str:
        del context, disambiguation, n
        if self._language == "zh_CN":
            return _ZH.get(source_text, source_text)
        return source_text


def tr(source: str) -> str:
    return QCoreApplication.translate("VideoCaptioner", source)


__all__ = ["CatalogTranslator", "tr"]
