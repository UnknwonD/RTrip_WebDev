    let currentStep = 1;

    function showStep(step) {
    document.querySelectorAll('.form-step').forEach((el) => {
        el.classList.remove('active');
    });

    const target = document.getElementById(`step${step}`);
    if (target) {
        target.classList.add('active');
        currentStep = step;

        updateProgress(step);

        if (step === 11) {
            analyzeStyles();  // step10에서 분석 실행
        }
    }
}
    function updateProgress(step) {
        const total = 10;
        for (let i = 1; i <= total; i++) {
            const el = document.getElementById(`progress-bar-${step}-${i}`);
            if (el) {
                el.classList.toggle('active', i <= step);
            }
        }

        // 숫자 레이블도 업데이트
        const label = document.getElementById(`progress-label-${step}`);
        if (label) {
            label.innerText = step;
        }
    }
    function analyzeStyles() {
        const labels = [
            ["자연형", "도시형", "자연 vs 도시", "blue"],
            ["숙박러", "당일러", "숙박 vs 당일", "orange"],
            ["새로운 지역파", "익숙한 지역파", "지역 성향", "green"],
            ["럭셔리파", "실속파", "숙소 성향", "pink"],
            ["휴양러", "체험러", "여행 목적", "purple"],
            ["숨은 장소 선호", "핫플 선호", "장소 취향", "red"],
            ["계획형", "즉흥형", "여행 스타일", "indigo"],
            ["사진 무관심", "사진 중요", "기억 방식", "teal"]
        ];

        let html = `<h4 class="text-center">📊 나의 여행 성향 분석</h4>`;

        for (let i = 0; i < 8; i++) {
            const score = scores[i];
            const percent = Math.round((score / 8) * 100);
            const leftPercent = 100 - percent;
            const [leftLabel, rightLabel, title, color] = labels[i];

            const isRightDominant = percent > 50;

            html += `
                <div class="style-bar">
                    <div class="style-bar-title">${title}</div>
                    <div class="style-bar-labels">
                        <span>${leftLabel} (${leftPercent}%)</span>
                        <span>${rightLabel} (${percent}%)</span>
                    </div>
                    <div class="style-bar-progress">
                        ${isRightDominant ? `
                            <div class="style-bar-fill right-fill" style="width: ${percent}%;" data-color="${color}"></div>
                        ` : `
                            <div class="style-bar-fill left-fill" style="width: ${leftPercent}%;" data-color="${color}"></div>
                        `}
                    </div>
                </div>
            `;
        }

        document.getElementById("style-analysis").innerHTML = html;
    }



    function nextStep() {
        showStep(currentStep + 1);
    }

    document.addEventListener("DOMContentLoaded", () => {
        let isDragging = false;
        let startX;
        let scrollLeft;

        const slider = document.getElementById("slider-container");
        const wrapper = document.getElementById("slider-wrapper");

        // Mouse events
        wrapper.addEventListener("mousedown", (e) => {
        isDragging = true;
        wrapper.classList.add("dragging");
        startX = e.pageX - wrapper.offsetLeft;
        scrollLeft = wrapper.scrollLeft;
        });

        wrapper.addEventListener("mouseleave", () => {
        isDragging = false;
        wrapper.classList.remove("dragging");
        });

        wrapper.addEventListener("mouseup", () => {
        isDragging = false;
        wrapper.classList.remove("dragging");
        });

        wrapper.addEventListener("mousemove", (e) => {
        if (!isDragging) return;
        e.preventDefault();
        const x = e.pageX - wrapper.offsetLeft;
        const walk = (x - startX) * 1.2; // drag 속도 조절
        wrapper.scrollLeft = scrollLeft - walk;
        });

        // Touch events
        wrapper.addEventListener("touchstart", (e) => {
        isDragging = true;
        startX = e.touches[0].pageX - wrapper.offsetLeft;
        scrollLeft = wrapper.scrollLeft;
        });

        wrapper.addEventListener("touchend", () => {
        isDragging = false;
        });

        wrapper.addEventListener("touchmove", (e) => {
        if (!isDragging) return;
        const x = e.touches[0].pageX - wrapper.offsetLeft;
        const walk = (x - startX) * 1.2;
        wrapper.scrollLeft = scrollLeft - walk;
        });
        showStep(currentStep);
    });

    // 가중치 매핑 테이블 
    const weightMap = {
        1: {
            1: [-1, 0, -1, 1, -1, 0, 1, 1],  // 해먹
            2: [1, 0, 1, 0, 1, 0, 0, 1]      // 롯데월드
        },
        2: {
            1: [1, 1, 0, -1, -1, 0, -1, 0],  // 호텔
            2: [-1, 1, 0, 1, 1, 0, 0, 0]     // 캠핑
        },
        3: {
            1: [1, 1, 1, 0, -1, 0, 1, -1],   // 백화점
            2: [0, -1, -1, 0, 1, 0, 0, 1]    // 쁘띠프랑스
        },
        4: {
            1: [-1, -1, 0, 1, 1, -1, -1, -1],// 계곡
            2: [1, 1, 1, 0, -1, 1, 1, 1]     // 반포대교
        },
        5: {
            1: [0, 0, -1, 0, 0, 1, 0, -1],   // 불국사
            2: [0, -1, 0, 1, -1, 0, 1, 0]    // 게스트하우스
        },
        6: {
            1: [-1, -1, -1, 0, 0, 1, -1, 1], // 전주 한옥마을
            2: [1, 1, 1, -1, 0, 1, 1, 1]     // 서울 야경
        },
        7: {
            1: [0, 0, -1, 0, 0, -1, 1, 1],   // 천안 미나릿길
            2: [-1, 0, -1, 0, 0, -1, -1, 1]  // 덕진공원
        },
        8: {
            1: [0, 0, 0, 1, 0, 0, 1, -1],    // 로데오거리
            2: [1, 0, 1, -1, 0, 0, 0, -1]    // 인계동
        },
        9: {
            1: [0, -1, 0, -1, -1, 1, -1, 0], // 여수
            2: [-1, 0, 0, 0, 1, -1, 1, -1]   // 마이산
        },
        10: {
            1: [1, 1, 0, 0, 1, -1, -1, -1],  // 안동하회마을
            2: [-1, -1, 1, -1, 0, 0, -1, -1] // 그랜드 조선 부산
        }
    };

    // 성향분석 이미지 계산 
    const scores = Array(8).fill(4);

    function selectStyle(step, choice) {
        const weight = weightMap[step][choice];
        for (let i = 0; i < 8; i++) {
            scores[i] += weight[i];
        }

        if (step === 10) {
            // 점수 정리
            for (let i = 0; i < 8; i++) {
                scores[i] = Math.max(1, Math.min(7, scores[i]));
            }

            // 서버에 점수 전송 (POST)
            fetch("/analyze_styles", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ scores }),
            })
            .then(response => response.json())
            .then(data => {
                // 이미지와 분석 결과 처리
                window.recommendedImages = data.images; // 예: [{url, area}]
                window.scores = scores; // 분석 결과 재사용

                // Step11 보여주고 분석 실행
                showStep(11);
                analyzeStyles();
                renderImages();
            });
        } else {
            nextStep();
        }
    }

// 이미지 결과 렌더링
function renderImages() {
    const container = document.getElementById("slider-container");
    container.innerHTML = "";

    (window.recommendedImages || []).forEach((img, index) => {
        container.innerHTML += `
            <div class="slide">
                <img src="${img.url}" alt="image ${index + 1}" class="img-responsive" style="width: 100%; object-fit: cover;" draggable="false">
                <div class="landing-text text-center">${img.area}</div>
            </div>
        `;
    });
}