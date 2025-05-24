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
        const userId = document.getElementById("USER_ID").value;
        fetch(`/check_duplicate?field=USER_ID&value=${userId}`)
            .then(res => res.json())
            .then(data => {
                const result = document.getElementById("id-check-result");
                result.innerText = data.duplicate 
                    ? "이미 사용 중인 아이디입니다." 
                    : "사용 가능한 아이디입니다.";
                result.classList.toggle("text-danger", data.duplicate);
                result.classList.toggle("text-success", !data.duplicate);
            })
            .catch(() => alert("중복 확인 중 오류가 발생했습니다."));
    }