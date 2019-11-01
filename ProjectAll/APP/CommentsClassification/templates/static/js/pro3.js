$(function(){
	// alert("1111");
	// 
	var sSaying = "";
	var temp = "";
	var mydict = {};
	var columns = ['交通是否便利(训练数据)',
				   '距离商圈远近(训练数据)',
				   "是否容易寻找(训练数据)",
				   '排队等候时间(训练数据)',
				   '服务人员态度(训练数据)',
				   '是否容易停车(训练数据)',
				   '点菜/上菜速度(训练数据)',
				   '价格水平(训练数据)',
				   '性价比(训练数据)',
				   '折扣力度(训练数据)',
				   '装修情况 (训练数据)',
				   '嘈杂情况(训练数据)',
				   '就餐空间(训练数据)',
				   '卫生情况(训练数据)',
				   '分量(训练数据)',
				   '口感(训练数据)',
				   '外观(训练数据)',
				   '推荐程度(训练数据)',
				   '本次消费感受(训练数据)',
				   '再次消费的意愿(训练数据)'];
	function freshdict(){
		mydict = {"name":"类别",
				  "children":
				  [
				    {
				      "name":"位置" ,
				      "children":
				      [
				        {"name":"交通是否便利" },
				        {"name":"距离商圈远近" },
				        {"name":"是否容易寻找" },
				      ]
				    },
				 
				    {
				      "name":"服务" ,
				      "children":
				      [
				 
				        {"name":"排队等候时间"},
				        {"name":"服务人员态度"},
				        {"name":"是否容易停车"},
				        {"name":"点菜/上菜速度"},
				      ]
				    },
				    {
				      "name":"价格" ,
				      "children":
				      [
				 
				        {"name":"价格水平"},
				        {"name":"性价比"},
				        {"name":"折扣力度"},
				      ]
				    },
				    {
				      "name":"环境" ,
				      "children":
				      [
				 
				        {"name":"装修情况"},
				        {"name":"嘈杂情况"},
				        {"name":"就餐空间"},
				        {"name":"卫生情况"},
				      ]
				    },
				    {
				      "name":"菜品" ,
				      "children":
				      [			
				        {"name":"分量"},
				        {"name":"口感"},
				        {"name":"外观"},
				        {"name":"推荐程度"},
				      ]
				    },
				    {
				      "name":"其他" ,
				      "children":
				      [
				 
				        {"name":"本次消费感受"},
				        {"name":"再次消费的意愿"},

				      ]
				    }			    				   
				  ]
				};
	};

	$(".nav-stacked > li").eq(1).click(function(){
		var cur = $(".nav-stacked > li .active").text();
		if (cur != "模型介绍"){
			$(".showpalce").html("");
			$(".nav-stacked > li").removeClass('active');
			$(this).addClass('active');

			temp = "<div class=\"row\"> \
		        <div class=\"col-lg-4\">\
		          <h3>TFIDF+NetWork</h3>\
		          <p class=\"text-danger\">优点：</p>\
		          <p class=\"text-danger\">缺点：</p>\
		          <p>该模型采用TFIDF来对评论数据进行表征，然后使用普通神经网络构建模型，该模型主要用于初步快速测试</p>\
		        </div>\
		        <div class=\"col-lg-4\">\
		          <h3>CNN+Attention</h3>\
		          <p class=\"text-danger\">优点：可以GPU加速，速度快</p>\
		          <p class=\"text-danger\">缺点：</p>\
		          <p>该模型构建Cnn网络进行分类，使用word embedding进行序列降维，由于输出是20个类别的分类，使用了self-atteion机制对Cnn卷积输出进行特征捕捉\
		          其实多复杂化网络结果能达到类似效果，但模型参数会过于臃肿，训练起来太慢。</p>\
		        </div>\
		        <div class=\"col-lg-4\">\
		          <h3>GRU+Attention</h3>\
		          <p class=\"text-danger\">优点：模型效果稍微提升</p>\
		          <p class=\"text-danger\">缺点：RNN层无法进行GPU加速，训练速度慢</p>\
		          <p>该模型类似模型2，只不过把原Cnn网络层替换成Rnn网络，然后使用self-atteion机制对Rnn输出序列进行特征捕捉\
		          该模型较Cnn模型效果有稍许提升，缺点是不能并行计算，训练时间相对较长</p>\
		        </div></div>\
		        \
		        <div class=\"row\"> \
		        <div class=\"col-lg-4\">\
		          <h3>BERT</h3>\
		          <p class=\"text-danger\">优点：</p>\
		          <p class=\"text-danger\">缺点：</p>\
		          <p>采用BERT模型进行分类</p>\
		        </div>\
		        <div class=\"col-lg-4\">\
		        </div>\
		        <div class=\"col-lg-4\">\
		        </div></div>";
			$(".showpalce").html(temp);}
		return false;
	})


	$(".nav-stacked > li").eq(0).click(function(){
		var cur = $(".nav-stacked > li .active").text();
		if (cur != "分类结果"){
			$(".showpalce").html("");
			$(".nav-stacked > li").removeClass('active');
			$(this).addClass('active');}	

			temp = "";
			temp = '<div class=\"row\"><div class=\"col-md-8\"id=\'mytree\'></div><div class=\"col-md-4\">';
			temp += "<div><table class=\"table\"><thead><tr><th> \
				情感倾向值</th><th>含义</th></tr></thead><tbody>"
			temp +="<tr><th scope=\"row\">"+'1'+"</th><td>"+'正面情感'+"</td></tr>" + 
				  	"<tr><th scope=\"row\">"+'0'+"</th><td>"+'中性情感'+"</td></tr>" +
				  	"<tr><th scope=\"row\">"+'-1'+"</th><td>"+'负面情感'+"</td></tr>" +
				  	"<tr><th scope=\"row\">"+'-2'+"</th><td>"+'未提及'+"</td></tr>"+
				  	"</tbody></table></div></div></div>";

			$(".showpalce").html(temp);
			if(d3.select('svg').length > 0){d3.select('svg').remove()};
			Graphshow(mydict);

		return false;
	})

	$(".nav-stacked > li").eq(2).click(function(){
		var cur = $(".nav-stacked > li .active").text();
		if (cur != "模型评估"){
			$(".showpalce").html("");
			$(".nav-stacked > li").removeClass('active');
			$(this).addClass('active');
			show_evaluate();
			$(".btn-group > button").click(function(){
				var data = {'id':$(this).text()};
				if(data['id'] != "类别:"){
					$('#mytitle').html(columns[parseInt(data['id'])-1]);
					$.ajax({
						url:'/CommentsClassification/getevaluate',
						type:'GET',
						cache: false,
						data:data,
						dataType:'json',
						success:function(result){
							$(".barchart").html('');
							$("#container").html('');
							$("#legend").html('');
							$("#dataView").html('');
							confusion_matric(result['metric'],result['score']);
							Barshow(result['barchart']);
						}
					})
				}
			});
			$(".btn-group > button").eq(1).click();
		}
		return false;
	})


	freshdict();
	$(".nav-stacked > li").eq(0).click();

	$("input").eq(0).click(function(){
		$.ajax({
			url:'/CommentsClassification/getcomments',
			type:'GET',
			cache: false,
			dataType:'json',
			success:function(result){
				$("textarea").val(result['text']);
				freshdict();
				var count = 0;
				for(var i in mydict['children']){
					for(var j in mydict['children'][i]['children']){
						mydict['children'][i]['children'][j]['name'] +=' :'+result['pred'][count];
						count +=1;
					};
				};
				$(".nav-stacked > li").eq(0).click();
			}
		})
	
	})


	function Graphshow(treeData){
		var margin = {top: 10, right: 30, bottom: 10, left:50},
	    width = 600 - margin.right - margin.left,
	    height = 500 - margin.top - margin.bottom;
		    
		var i = 0,duration = 750,root;
		//定义数据转换函数
		var tree = d3.layout.tree()
		    .size([height, width]);
		//定义对角线生成器diagonal
		var diagonal = d3.svg.diagonal()
		    .projection(function(d) { return [d.y, d.x]; });
		//定义svg
		var svg = d3.select("#mytree").append("svg")
			.attr("width", width + margin.right + margin.left)
			.attr("height", height + margin.top + margin.bottom)
			.append("g")
			.attr("transform", "translate(" + margin.left + "," + margin.top + ")");

		root = treeData;
		root.x0 = height / 2;
		root.y0 = 0;

		update(root);
		d3.select(self.frameElement).style("height", "500px");


		function update(source) {
		  // Compute the new tree layout.
		  var nodes = tree.nodes(root).reverse(),
		      links = tree.links(nodes);

		  // Normalize for fixed-depth.控制横向深度
		  nodes.forEach(function(d) { d.y = d.depth * 150; });

		  // Update the nodes
		  var node = svg.selectAll("g.node")
		      .data(nodes, function(d) { return d.id || (d.id = ++i);});

		  // Enter any new nodes at the parent's previous position.
		  var nodeEnter = node.enter().append("g")
		      .attr("class", "node")
		      .attr("transform", function(d) { return "translate(" + source.y0 + "," + source.x0 + ")"; })
		      .on("click", click);

		  nodeEnter.append("circle")
		      .attr("r", 1e-6)
		      .style("fill", function(d) { return d._children ? "#ccff99" : "#fff"; });

		  nodeEnter.append("text")
		      .attr("x", function(d) { return d.children || d._children ? -13 : 13; })
		      .attr("dy", ".35em")
		      .attr("text-anchor", function(d) { return d.children || d._children ? "end" : "start"; })
		      .text(function(d) { return d.name; })
		      .style("fill-opacity", 1e-6)
		     .attr("class", function(d) {
		              if (d.url != null) { return 'hyper'; } 
		         })
		          .on("click", function (d) { 
		              $('.hyper').attr('style', 'font-weight:normal');
		              d3.select(this).attr('style', 'font-weight:bold');
		              if (d.url != null) {
		                 //  window.location=d.url; 
		                 $('#vid').remove();

		                 $('#vid-container').append( $('<embed>')
		                    .attr('id', 'vid')
		                    .attr('src', d.url + "?version=3&amp;hl=en_US&amp;rel=0&amp;autohide=1&amp;autoplay=1")
		                    .attr('wmode',"transparent")
		                    .attr('type',"application/x-shockwave-flash")
		                    .attr('width',"100%")
		                    .attr('height',"100%") 
		                    .attr('allowfullscreen',"true")
		                    .attr('title',d.name)
		                  )
		                }
		          });

		  // Transition nodes to their new position.
		  var nodeUpdate = node.transition()
		      .duration(duration)
		      .attr("transform", function(d) { return "translate(" + d.y + "," + d.x + ")"; });

		  nodeUpdate.select("circle")
		      .attr("r", 6)
		      .style("fill", function(d) { return d._children ? "#ccff99" : "#fff"; });

		  nodeUpdate.select("text")
		      .style("fill-opacity", 1);

		  // Transition exiting nodes to the parent's new position.
		  var nodeExit = node.exit().transition()
		      .duration(duration)
		      .attr("transform", function(d) { return "translate(" + source.y + "," + source.x + ")"; })
		      .remove();

		  nodeExit.select("circle")
		      .attr("r", 1e-6);

		  nodeExit.select("text")
		      .style("fill-opacity", 1e-6);

		  // Update the links鈥�
		  var link = svg.selectAll("path.link")
		      .data(links, function(d) { return d.target.id; });

		  // Enter any new links at the parent's previous position.
		  link.enter().insert("path", "g")
		      .attr("class", "link")
		      .attr("d", function(d) {
		        var o = {x: source.x0, y: source.y0};
		        return diagonal({source: o, target: o});
		      });

		  // Transition links to their new position.
		  link.transition()
		      .duration(duration)
		      .attr("d", diagonal);

		  // Transition exiting nodes to the parent's new position.
		  link.exit().transition()
		      .duration(duration)
		      .attr("d", function(d) {
		        var o = {x: source.x, y: source.y};
		        return diagonal({source: o, target: o});
		      })
		      .remove();

		  // Stash the old positions for transition.
		  nodes.forEach(function(d) {
		    d.x0 = d.x;
		    d.y0 = d.y;
		  });
		}

		// Toggle children on click.
		function click(d) {
		  if (d.children) {
		    d._children = d.children;
		    d.children = null;
		  } else {
		    d.children = d._children;
		    d._children = null;
		  }
		  update(d);
		}
	}

	var margin = {top: 10, right: 10, bottom: 50, left: 20};
	function Matrix(options) {
	    var width = 280,
	    height = 280,
	    data = options.data,
	    container = options.container,
	    labelsData = options.labels,
	    startColor = options.start_color,
	    endColor = options.end_color;

		var widthLegend = 80;
		if(!data){
			throw new Error('Please pass data');
		}

		if(!Array.isArray(data) || !data.length || !Array.isArray(data[0])){
			throw new Error('It should be a 2-D array');
		}

	    var maxValue = d3.max(data, function(layer) { return d3.max(layer, function(d) { return d; }); });
	    var minValue = d3.min(data, function(layer) { return d3.min(layer, function(d) { return d; }); });

		var numrows = data.length;
		var numcols = data[0].length;

		var svg = d3.select(container).append("svg")
		    .attr("width", width + margin.left + margin.right)
		    .attr("height", height + margin.top + margin.bottom)
			.append("g")
		    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

		var background = svg.append("rect")
		    .style("stroke", "black")
		    .style("stroke-width", "2px")
		    .attr("width", width)
		    .attr("height", height);

		var x = d3.scale.ordinal()
		    .domain(d3.range(numcols))
		    .rangeBands([0, width]);

		var y = d3.scale.ordinal()
		    .domain(d3.range(numrows))
		    .rangeBands([0, height]);

		var colorMap = d3.scale.linear()
		    .domain([minValue,maxValue])
		    .range([startColor, endColor]);

		var row = svg.selectAll(".row")
		    .data(data)
		  	.enter().append("g")
		    .attr("class", "row")
		    .attr("transform", function(d, i) { return "translate(0," + y(i) + ")"; });

		var cell = row.selectAll(".cell")
		    .data(function(d) { return d; })
				.enter().append("g")
		    .attr("class", "cell")
		    .attr("transform", function(d, i) { return "translate(" + x(i) + ", 0)"; });

		cell.append('rect')
		    .attr("width", x.rangeBand())
		    .attr("height", y.rangeBand())
		    .style("stroke-width", 0);

	    cell.append("text")
		    .attr("dy", ".32em")
		    .attr("x", x.rangeBand() / 2)
		    .attr("y", y.rangeBand() / 2)
		    .attr("text-anchor", "middle")
		    .style("fill", function(d, i) { return d >= maxValue/2 ? 'white' : 'black'; })
		    .text(function(d, i) { return d; });

		row.selectAll(".cell")
		    .data(function(d, i) { return data[i]; })
		    .style("fill", colorMap);

		var labels = svg.append('g')
			.attr('class', "labels");

		var columnLabels = labels.selectAll(".column-label")
		    .data(labelsData)
		    .enter().append("g")
		    .attr("class", "column-label")
		    .attr("transform", function(d, i) { return "translate(" + x(i) + "," + height + ")"; });

		columnLabels.append("line")
			.style("stroke", "black")
		    .style("stroke-width", "1px")
		    .attr("x1", x.rangeBand() / 2)
		    .attr("x2", x.rangeBand() / 2)
		    .attr("y1", 0)
		    .attr("y2", 5);

		columnLabels.append("text")
		    .attr("x", 45)
		    .attr("y", y.rangeBand() / 2 - 20)
		    .attr("dy", ".22em")
		    .attr("text-anchor", "end")
		    // .attr("transform", "rotate(-60)")
		    .text(function(d, i) { return d; });

		var rowLabels = labels.selectAll(".row-label")
		    .data(labelsData)
		  .enter().append("g")
		    .attr("class", "row-label")
		    .attr("transform", function(d, i) { return "translate(" + 0 + "," + y(i) + ")"; });

		rowLabels.append("line")
			.style("stroke", "black")
		    .style("stroke-width", "1px")
		    .attr("x1", 0)
		    .attr("x2", -5)
		    .attr("y1", y.rangeBand() / 2)
		    .attr("y2", y.rangeBand() / 2);

		rowLabels.append("text")
		    .attr("x", -8)
		    .attr("y", y.rangeBand() / 2)
		    .attr("dy", ".32em")
		    .attr("text-anchor", "end")
		    .text(function(d, i) { return d; });

	    var key = d3.select("#legend")
	    .append("svg")
	    .attr("width", widthLegend)
	    .attr("height", height + margin.top + margin.bottom);

	    var legend = key
	    .append("defs")
	    .append("svg:linearGradient")
	    .attr("id", "gradient")
	    .attr("x1", "100%")
	    .attr("y1", "0%")
	    .attr("x2", "100%")
	    .attr("y2", "100%")
	    .attr("spreadMethod", "pad");

	    legend
	    .append("stop")
	    .attr("offset", "0%")
	    .attr("stop-color", endColor)
	    .attr("stop-opacity", 1);

	    legend
	    .append("stop")
	    .attr("offset", "100%")
	    .attr("stop-color", startColor)
	    .attr("stop-opacity", 1);

	    key.append("rect")
	    .attr("width", widthLegend/2-10)
	    .attr("height", height)
	    .style("fill", "url(#gradient)")
	    .attr("transform", "translate(0," + margin.top + ")");

	    var y = d3.scale.linear()
	    .range([height, 0])
	    .domain([minValue, maxValue]);

	    var yAxis = d3.svg.axis()
	    .scale(y)
	    .orient("right");

	    key.append("g")
	    .attr("class", "y axis")
	    .attr("transform", "translate(31," + margin.top + ")")
	    .call(yAxis)

	}

	function confusion_matric(confusionMatrix, computedData){
		//confusionMatrix, computedData
		// var confusionMatrix = [
		// 	[169, 10,8,1],
		// 	[7, 46,10,10],
		// 	[12,13,50,20],
		// 	[12,13,50,20]
		// ];

        var labels = ['-2', '-1','0','1'];
		Matrix({
			container : '#container',
			data      : confusionMatrix,
			labels    : labels,
            start_color : '#ffffff',
            end_color : '#e67e22'
		});
		// var computedData = [{"F1":0.9, "PRECSION":0.92,"RECALL":0.94,"ACC":0.93}];
		var table = tabulate(computedData, ["F1", "PRECSION","RECALL","ACC"]);
	}


	function tabulate(data, columns) {
	    var table = d3.select("#dataView").append("table")
	            .attr("class", "table")
	            .attr("style",'width:90%'),
	        thead = table.append("thead"),
	        tbody = table.append("tbody");

	    // append the header row
	    thead.append("tr")
	        .selectAll("th")
	        .data(columns)
	        .enter()
	        .append("th")
	            .text(function(column) { return column; });

	    // create a row for each object in the data
	    var rows = tbody.selectAll("tr")
	        .data(data)
	        .enter()
	        .append("tr");

	    // create a cell in each row for each column
	    var cells = rows.selectAll("td")
	        .data(function(row) {
	            return columns.map(function(column) {
	                return {column: column, value: row[column]};
	            });
	        })
	        .enter()
	        .append("td")
	        .attr("style", "font-family: Courier") // sets the font style
	            .html(function(d) { return d.value; });

	    return table;
	}


	function Barshow(data){
            var datax = ['-2', '-1', '0', '1'];
            var datay = [0, 0, 0, 0];
            // var data = [{'name':'-2','value':120},{'name':'-1','value':222}, 
           	// 			 {'name':'0','value':150},{'name':'1','value':80}];
           	datay[0] = data[0]['value'];
           	datay[1] = data[1]['value'];
           	datay[2] = data[2]['value'];
           	datay[3] = data[3]['value'];

			var margin = {top: 20, right: 40, bottom: 30, left: 50},
			    width = 300 - margin.left - margin.right,
			    height = 400 - margin.top - margin.bottom;

			var x = d3.scale.ordinal()
			    .rangeRoundBands([0, width], .1);

			var y = d3.scale.linear()
			    .range([height, 0]);

			var xAxis = d3.svg.axis()
			    .scale(x)
			    .orient("bottom");

			var yAxis = d3.svg.axis()
			    .scale(y)
			    .orient("left");

			var chart = d3.select(".barchart")
			    .attr("width", width + margin.left + margin.right)
			    .attr("height", height + margin.top + margin.bottom)
			  	.append("g")
			    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");
	 
	 		x.domain(datax);
  			y.domain([0, d3.max(datay)]);

			chart.append("g")
			  .attr("class", "x axis")
			  .attr("transform", "translate(0," + height + ")")
			  .call(xAxis);

			chart.append("g")
			  .attr("class", "y axis")
			  .call(yAxis);

			chart.selectAll(".bar")
			  .data(data)
			  .enter().append("rect")
				  .attr("class", "bar")
				  .attr("x", function(d) { return x(d.name); })
				  .attr("y", function(d) { return y(d.value); })
				  .attr("height", function(d) { return height - y(d.value); })
				  .attr("width", x.rangeBand());

	}

	function show_evaluate(){

	   temp = "<div class=\"btn-toolbar\" role=\"toolbar\" aria-label=\"Toolbar with button groups\">\
      <div class=\"btn-group\" role=\"group\" aria-label=\"First group\">\
        <button type=\"button\" class=\"btn btn-default\">类别:</button>\
        <button type=\"button\" class=\"btn btn-default\">1</button>\
        <button type=\"button\" class=\"btn btn-default\">2</button>\
        <button type=\"button\" class=\"btn btn-default\">3</button>\
        <button type=\"button\" class=\"btn btn-default\">4</button>\
        <button type=\"button\" class=\"btn btn-default\">5</button>\
        <button type=\"button\" class=\"btn btn-default\">6</button>\
        <button type=\"button\" class=\"btn btn-default\">7</button>\
        <button type=\"button\" class=\"btn btn-default\">8</button>\
        <button type=\"button\" class=\"btn btn-default\">9</button>\
        <button type=\"button\" class=\"btn btn-default\">10</button>\
        <button type=\"button\" class=\"btn btn-default\">11</button>\
        <button type=\"button\" class=\"btn btn-default\">12</button>\
        <button type=\"button\" class=\"btn btn-default\">13</button>\
        <button type=\"button\" class=\"btn btn-default\">14</button>\
        <button type=\"button\" class=\"btn btn-default\">15</button>\
        <button type=\"button\" class=\"btn btn-default\">16</button>\
        <button type=\"button\" class=\"btn btn-default\">17</button>\
        <button type=\"button\" class=\"btn btn-default\">18</button>\
        <button type=\"button\" class=\"btn btn-default\">19</button>\
        <button type=\"button\" class=\"btn btn-default\">20</button>\
		</div></div>"

       // <h4 style=\"text-align: center;\">这是该类在测试集上的表现</h4>
		temp += "<div class=\"row\"><div class=\"col-md-6\"><br></div>\
					<div class=\"col-md-6\"><br></div>\
			    </div>\
				<div class=\"row\">\
					<div class=\"col-md-6\">\
					    <div><div id=\"dataView\"></div>\
					    <p style=\"margin: 0px 0px 0px 80px;\">Test DataSet Confusion Matric</p>\
						<div style=\"display:inline-block; float:left\" id=\"container\"></div>\
	    				<div style=\"display:inline-block; float:left\" id=\"legend\"></div></div>\
					</div>\
					<div class=\"col-md-6\"><p style=\"margin: 22px 0px 0px 85px;\"\
					id=\"mytitle\">Train DataSet</p>\
					<svg class=\"barchart\"></svg>\
					</div>\
				</div>";
		$(".showpalce").html(temp);
	}
})
