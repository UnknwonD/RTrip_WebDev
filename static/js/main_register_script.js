    let currentStep = 1;

    function showStep(step) {
        document.querySelectorAll('.form-step').forEach((el) => {
            el.classList.remove('active');
        });
        const target = document.getElementById(`step${step}`);
        if (target) {
            target.classList.add('active');
            currentStep = step;
        }
    }

    function nextStep() {
        showStep(currentStep + 1);
    }

    function selectStyle(step, value) {
        document.getElementById(`slider${step}`).value = value;
        nextStep();
    }

    document.addEventListener("DOMContentLoaded", () => {
        showStep(currentStep);
    });

function checkDuplicateId() {
    // step1이 보이는지 먼저 확인
    const step1 = document.getElementById('step1');
    if (!step1.classList.contains('active')) {
        showStep(1);
    }
    
    const userIdInput = document.getElementById("REGISTER_USER_ID");
    
    if (!userIdInput) {
        alert("아이디 입력란을 찾을 수 없습니다.");
        return;
    }
    
    const userId = userIdInput.value.trim();
    console.log("[DEBUG] 중복확인 실행 - 입력값:", userId);
    
    if (!userId) {
        alert("아이디를 입력해주세요.");
        return;
    }
    
    fetch(`/check_duplicate?field=USER_ID&value=${encodeURIComponent(userId)}`)
        .then(res => {
            console.log("[DEBUG] 서버 응답 상태:", res.status);
            return res.json();
        })
        .then(data => {
            console.log("[DEBUG] 서버 응답 데이터:", data);
            const result = document.getElementById("id-check-result");
            
            if (result) {
                result.innerText = data.duplicate
                    ? "이미 사용 중인 아이디입니다."
                    : "사용 가능한 아이디입니다.";
                result.classList.remove("text-danger", "text-success");
                result.classList.add(data.duplicate ? "text-danger" : "text-success");
                console.log("[DEBUG] 결과 표시됨:", result.innerText);
            } else {
                console.log("[DEBUG] 결과 표시 요소를 찾을 수 없음");
            }
        })
        .catch(error => {
            console.error("[DEBUG] 중복 확인 오류:", error);
            alert("중복 확인 중 오류가 발생했습니다.");
        });
}