function explore_text(t){

    if (t['word'] === "sheetz"){
	display_me = "Example reviews: <ul><li>It saddens me to have to give a Sheetz a one star review.  But I have to be honest.  Of all the Sheetz I've been too, and it's a lot, this one deserves the one star.\n\nLet's start with this placement of a store, horrible.  Cramming everything in there, including a car wash.  Which leads into number two complaint.  The wait...I waited ten minutes just to get through the check out line.  This can lead into downfall number three, the staff.  One person to man the registers while everyone else is in the kitchen preparing the food (ugh, we will get to that in a minute).  The staff reminded me of a bunch of frat brothers bragging about their prior night's achievements, or a group of old ladies complaining about everything.  I could probably recite to you their thoughts on a certain co-worker word for word, but I won't.\n\nNow, let's move on, let us visit my food experience here.  I ordered my food (number 90), proceeded to the cooler to select my beverage, then went to the front to pay (ten minutes, remember? and I was the fourth person in line).  I then waited another 25 minutes for my food to be prepared.  By the time I left the store, Number 117 was receiving their food.  How do I know that a person who ordered 27th after me was being called to receive their food.  Because when they finally called my number I received 117's food.  If I hadn't looked into the bag when I reached my car I would have probably been throwing away my food.  \n\nI walked back in and told them it wasn't right, which they didn't believe me.  I opened the bag, showed them the tags on the food, number 117, and then showed them my receipt, number 90.  How, might one ask, did such a mistake occur?  Well, because my food, when made, was not labelled, and it sat there for 25 minutes until they realized it was made.\n\nI would like to apologize to Number 117, if by chance you ever come across this review and read it.  I apologize that in my anger when I came back in to get my appropriate items I ordered, that I slammed your bag full of food back down on the counter in front of not only the employees, but you.  I hope your burger or what not you ordered was not smushed so much.\n\nAnger and the urge for Sheetz does not mix well for me.</li><li>Slow service and the food isn't as good as the Sheetz back in central PA where I was first introduced to them. Even though some nights there will be 3-4 behind the counter making food, it still seems to take forever for them to get MTO'S out. Maybe it's the set up of their kitchen, because I've never really had that problem anywhere else.\n\nAdding a major big shiny star for the always spotless bathrooms. It is seriously nice to know that in a busy business district like Monroeville there is at least one place that has a decent and well lit bathroom.\n\nThis location also has a dining area that would probably be good for travelers who need a break from the road. If I recall correctly, I'm pretty sure there is also a walk in pop cooler cave. Isn't that just all the way Pittsburgh for ya?\n\nOh, and always get the boom boom sauce.</li><li></li></ul>";
    }else if (t['word'] === 'friendly'){
	display_me = "Example reviews: <ul><li>One of the better GetGos around, but nothing to write home about.  Unlike most locations, they didn't have a mentally challenged civil engineer/architect laying out this one: it's easy to get in and out, no bad congestion, no underutilized space.  The clerks are also usually pretty friendly, which is unusual for a GetGo.\n\nThis is a Get Go-wide problem: these guys have the SLOWEST GASOLINE PUMPS in the world.  Seriously.  It takes me at least twice as long to get gas at Get Go than it does at other gas stations.  The computer takes forever to process inputs (card swipes, credit/debit?, car wash?, etc.), and they pump gas about 2/3 as fast as other stations.  This partly explains the lines at Get Go stations.  I've even taken to hitting a less-convenient but similarly priced BP near work just because it's much easier to get in and out.  (Because let's be honest: getting gas is not a \"fun\" thing to do.  Unless you're at a Wawa or a QT.  Why won't Wawa cross the mountains and come to Pittsburgh?)</li><li>No complaints here.  Very friendly staff.</li><li>I'm saddened by my need to venture further than a block away to purchase items that one may typically find at drug stores. While I frequent this establishment due to it's proximity to my home and GREATLY appreciate that they carry Pepsi and Mountain Dew THROWBACK, their selection of typical drug store goods can sometimes be disappointing. \n\nWhile I know that not everyone wants Arm and Hammer naturals unscented deodorant or Wax Strips for body hair removal or women's electric shavers, I have come up short searching for all of these items in this location, making me head to Oakland's CVS where the young college kids must buy more hair removal products and hippy deodorants. \n\n Service is very friendly but typically snail-paced. And finally, when I say I don't need a plastic bag I honestly mean it, sweet old cashier lady. For realz. I have two hands that are remarkably useful for carrying things. Though I do appreciate your concern :) \n\nThe three stars are given for convenience of location, carrying Pepsi and Mt Dew Throwback (HFCS is whack), and the ability to get $100 cash back per purchase and allowing local residents to use your parking lot as a through-street due to the convoluted design of the immediate area; for these things I thank you.</li></ul>";
    }else if (t['word'] === 'beware'){
	display_me = "Example reviews: <ul><li>Buyer Beware.  Don't stop at this location for coffee in the morning.  Three times I have tried in the past month and three times I have been disappointed.  Twice all the carafes had \"expired\" so you either had to wait for a fresh brew or drink burnt coffee.  And the third time, all carafes were out ... entirely out.  The rest of the store is good.  Employees are nice.  Food tastes great.  Donuts are always fresh.  And during the lunch rush, you can usually get in and out pretty fast.  Despite all those positives, there's better options until they sort out their coffee.</li><li>Was there late this morning. There is a guy Jason in the kitchen area that is ignorant, rude, and just plain out disgraceful. The way he talk to the customer in front of us and the way he talks to the employees is just astonishing. When I asked who his boss is, cause I was not happy with the way he was. He told me \"to leave his store. He is in charge and to get out\". I said I asked you for your bosses name. He screamed get out now. Never again will I step foot in this location. GetGo needs to rethink what they stand for. Customer service is horrible in this store and I will be filling a complaint with the higher ups. Beware and take your business elsewhere if he is allowed to do this stuff to people. Has anyone else dealt with him?</li></ul>";
    }
    
    var sect = document.getElementById("doubleclick");
    sect.innerHTML = display_me;
}

function draw_chart(data, selection, section2){

    var total = d3.sum(data, function(d, i) { return d[selection]; });
    // Sort the data in descending order
    data.sort(function(a, b) {return d3.descending(a[selection], b[selection])});
      
    // Setup the scale for the values for display, use abs max as max value
    var x = d3.scale.linear()
        .domain([0, d3.max(data, function(d) { return Math.abs(d[selection]); })])
        .range(["0%", "100%"]);
    
      
    var table = d3.select("#resultstable");
    table.selectAll("tr").remove();
      
    // Create a table with rows and bind a data row to each table row
    var tr = table.selectAll("tr.data")
        .data(data.filter(function(d) { return d[selection] != 0; }))
        .enter()
        .append("tr")
        .attr("class", "datarow");

    var max = 20;
    // Set the even rows
    d3.selectAll(".datarow").filter(":nth-child(even)").attr("class", "datarow even");

    //hide zero rows?
    
    
    // Create the name column
    tr.append("td").attr("class", "data name")
        .text(function(d) { return d['word'] })        
	.attr("class", function(d) { return  isdoubleclickable(d['word']) ? "blueclick" : ""; })
        .on('click', function(d,i){ explore_text(d);  });;
    
    ////////////////////////////////////////////
    //// column 1
    // Create the percent value column
    tr.append("td").attr("class", "data value")
        .html(function(d) { return "<div style='background: rgb(198, 2, 48); float: left; width: 100px; width: "+(d[selection].toFixed(4)*-5)+"px'>&nbsp;</div><div style='float:left; width: 20px; padding-left: 10px;'>"+d[selection].toFixed(4)+"</div><div style='clear:both;'></div>"; });
    
    // Creates the positive div bar
    /*tr.select("div.positive")
        .style("width", "0%")
        .transition()
        .duration(500)
        .style("width", function(d) { return d[selection] > 0 ? x(d[selection]) : "0%"; });
	*/
    ///////////////////////////////////////////
    //// column 2
    // Create the percent value column
    tr.append("td").attr("class", "data value")
        .html(function(d) { return "<div style='background: rgb(249, 166, 26); float: left; width: 100px; width: "+(d[section2].toFixed(4)*-5)+"px'>&nbsp;</div><div style='float:left; width: 20px; padding-left: 10px;'>"+d[section2].toFixed(4)+"</div><div style='clear:both;'></div>"; });

    
}

function isdoubleclickable(s){
    return s == "sheetz" || s == "beware" || s == "friendly";
}
