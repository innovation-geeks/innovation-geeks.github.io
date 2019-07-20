/*
  Tab 'Categories' expand/close effect.
*/
$(function() {
  var child_prefix = "l_";
  var parent_prefix = "h_";

  $(".collapse").on("hide.bs.collapse", function() { // Bootstrap collapse events.
    var parent_id = parent_prefix + $(this).attr('id').substring(child_prefix.length);
    if (parent_id) {
      $("#" + parent_id + " .far.fa-folder-open").attr("class", "far fa-folder fa-fw");
      $("#" + parent_id + " .category-trigger").addClass("flip");
      $("#" + parent_id).removeClass("hide-border-bottom");
    }
  });

  $(".collapse").on("show.bs.collapse", function() {
    var parent_id = parent_prefix + $(this).attr('id').substring(child_prefix.length);
    if (parent_id) {
      $("#" + parent_id + " .far.fa-folder").attr("class", "far fa-folder-open fa-fw");
      $("#" + parent_id + " .category-trigger").removeClass("flip");
      $("#" + parent_id).addClass("hide-border-bottom");
    }
  });

});