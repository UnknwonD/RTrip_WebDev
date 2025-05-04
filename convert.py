import re

def convert_html_for_flask(html_path, output_path):
    with open(html_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # 정규식으로 src/href 경로를 Flask 방식으로 변환
    pattern = r'(?P<attr>src|href)=["\'](?P<path>(css|js|images|fonts)/[^"\']+)["\']'
    
    def replace(match):
        attr = match.group("attr")
        path = match.group("path")
        return f'{attr}="{{{{ url_for(\'static\', filename=\'{path}\') }}}}"'

    updated_content = re.sub(pattern, replace, content)

    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(updated_content)
    
    print(f"✔ Flask HTML로 저장 완료: {output_path}")

# convert_html_for_flask("./templates/index.html", "index.html")
# convert_html_for_flask("./templates/contactthanks.html", "contactthanks.html")
# convert_html_for_flask("./templates/privacy.html", "privacy.html")
# convert_html_for_flask("./templates/shortcodes.html", "shortcodes.html")

# convert_html_for_flask("./templates/subscribe.html", "subscribe.html")
#  convert_html_for_flask("./templates/video.html", "video.html")

convert_html_for_flask("./templates/download.html", "download.html")

