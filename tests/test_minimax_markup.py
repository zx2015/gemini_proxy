from app.utils.minimax_tool_markup import recover_minimax_tool_calls_from_text

def test_recover_minimax_tool_calls_normal():
    """正常格式的 tool_call 恢复"""
    text = (
        "我来调研一下：]<]minimax[>[<tool_call>\n"
        "]<]minimax[>[<invoke name=\"list_files\">]<]minimax[>[<directory>/media/data/git/tubehub]<]minimax[>[</directory>]<]minimax[>[</invoke>\n"
        "]<]minimax[>[</tool_call>"
    )
    clean, calls = recover_minimax_tool_calls_from_text(text)
    assert clean.strip() == "我来调研一下："
    assert len(calls) == 1
    assert calls[0]["name"] == "list_files"
    assert calls[0]["args"] == {"directory": "/media/data/git/tubehub"}

def test_recover_minimax_tool_calls_malformed_quote():
    """缺失闭合双引号的极弱格式 (name=\"list_files) 恢复测试"""
    text = (
        "我来为您分析：]<]minimax[>[<tool_call>\n"
        "]<]minimax[>[<invoke name=\"list_files>]<]minimax[>[<directory>/media/data/git/tubehub]<]minimax[>[</directory>]<]minimax[>[</invoke>\n"
        "]<]minimax[>[</tool_call>"
    )
    clean, calls = recover_minimax_tool_calls_from_text(text)
    assert clean.strip() == "我来为您分析："
    assert len(calls) == 1
    assert calls[0]["name"] == "list_files"
    assert calls[0]["args"] == {"directory": "/media/data/git/tubehub"}

def test_recover_minimax_tool_calls_no_quotes():
    """完全没有双引号的格式 (name=list_files) 恢复测试"""
    text = (
        "分析如下：]<]minimax[>[<tool_call>\n"
        "]<]minimax[>[<invoke name=list_files>]<]minimax[>[<directory>/media/data/git/tubehub]<]minimax[>[</directory>]<]minimax[>[</invoke>\n"
        "]<]minimax[>[</tool_call>"
    )
    clean, calls = recover_minimax_tool_calls_from_text(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "list_files"
    assert calls[0]["args"] == {"directory": "/media/data/git/tubehub"}
