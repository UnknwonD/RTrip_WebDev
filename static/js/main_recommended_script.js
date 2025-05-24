document.querySelectorAll('.mission-button').forEach(btn => {
      btn.addEventListener('click', function () {
        this.classList.toggle('selected');
      });
    });
    
    document.getElementById('confirmMission').addEventListener('click', function () {
      const selected = document.querySelectorAll('.mission-button.selected');
      const values = Array.from(selected).map(el => el.getAttribute('data-code'));
      const labels = Array.from(selected).map(el => el.textContent.trim());
      
      const petChecked = document.getElementById('pet_checkbox').checked;
      if (petChecked) {
        values.unshift("0"); // 미션 코드 0으로 정의
        labels.unshift("🐾 반려동물 동반 여행");
      }
      document.getElementById('mission_ENC_hidden').value = values.join(',');
      document.getElementById('selected_missions_text').innerText =
        labels.length > 0 ? `선택한 미션: ${labels.join(', ')}` : '선택된 미션 없음';
    
      $('#missionModal').modal('hide');
    });

    const picker = new Litepicker({
        element: document.getElementById('date_range'),
        singleMode: false,
        format: 'YYYY-MM-DD',
        onSelect: (start, end) => {
            document.getElementById('start_date').value = start.format('YYYY-MM-DD');
            document.getElementById('end_date').value = end.format('YYYY-MM-DD');
        }
    });

    document.getElementById('missionToggle').addEventListener('change', function () {
        const isChecked = this.checked;
        const missionLabel = document.getElementById('missionToggleLabel');
        const missionFields = document.getElementById('normal_mission_fields');
        const missionTypeHidden = document.getElementById('mission_type_hidden');

        if (isChecked) {
            missionLabel.textContent = '무계획 여행';
            missionFields.style.display = 'none';
            missionTypeHidden.value = 'special';
        } else {
            missionLabel.textContent = '테마 여행';
            missionFields.style.display = 'block';
            missionTypeHidden.value = 'normal';
        }
    });

    // 초기화 시 미션 필드 보이도록
    document.getElementById('normal_mission_fields').style.display = 'block';


    document.querySelectorAll('.dislike-btn').forEach(btn => {
    btn.addEventListener('click', function () {
      const planIdx = this.dataset.plan;
      const spotIdx = this.dataset.spot;
      console.log(`플랜 ${planIdx}의 장소 ${spotIdx} 싫어요 → 대체 장소 요청`);
      $('#recommendation-carousel').carousel('next');
    });
  });

  document.querySelectorAll('.like-btn').forEach(btn => {
    btn.addEventListener('click', function () {
      const planIdx = this.dataset.plan;
      const spotIdx = this.dataset.spot;
      console.log(`플랜 ${planIdx}의 장소 ${spotIdx} 좋아요 저장`);
    });
  });

    function updateTotalCost() {
        const lodging = parseInt(document.getElementById("LODGOUT_COST").value) || 0;
        const activity = parseInt(document.getElementById("ACTIVITY_COST").value) || 0;
        document.getElementById("TOTAL_COST").value = lodging + activity;
    }

    document.getElementById("LODGOUT_COST").addEventListener("change", updateTotalCost);
    document.getElementById("ACTIVITY_COST").addEventListener("change", updateTotalCost);

    // 초기화 시 총합 계산
    updateTotalCost();

  $(document).ready(function () {
    const scrollLockTop = $('#recommendation-carousel').offset().top;

    // 왼쪽 버튼 클릭 시
    $('.left.carousel-control').on('click', function (e) {
      e.preventDefault();
      $('#recommendation-carousel').carousel('prev');
    });

    // 오른쪽 버튼 클릭 시
    $('.right.carousel-control').on('click', function (e) {
      e.preventDefault();
      $('#recommendation-carousel').carousel('next');
    });
  });