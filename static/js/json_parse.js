function JsonParsing() {
	// "C065MK78" - example auto nbr

	$.getJSON("/json", function(json) {
		console.log(json);
		
		var form = document.getElementById("form_response");
		var children = form.children;

		i = 0;
		for (key in json) {
			if (typeof children[i] != 'undefined') {
				children[i].children[0].append(json[key]);
			}
			++i;
		}
	});
}

/*
распознанный государственный регистрационный номер (ГРЗ)
распознанный номер региона ГРЗ
марка и модель транспортного средства
тип транспортного средства
идентификатор запроса
*/

JsonParsing();
