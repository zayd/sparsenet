#! sim -f

#
# init network
#
initSim 64 64

#
# init canvases
#
frame .buttons
frame .sliders

#
# init buttons
#
button .b_quit -text "quit" -command exit
button .b_go -text "go" -command {set run_p 1}
button .b_stop -text "stop" -command {set run_p 0}
button .b_step -text "step" -command stepNet
button .b_load -text "load" -command loadState
button .b_save -text "save" -command saveState

#
# init sliders
#
scale .beta -label beta -from 0 -to 50 -orient horizontal \
	-command {setVar beta}
.beta set 14

scale .eta -label eta -from 0 -to 500 -orient horizontal \
	-command {setVar eta}
.eta set 500

scale .settle -label settle -from 0 -to 50 -orient horizontal \
	-command {setVar settle}
.settle set 10

#
# pack
#
pack .buttons .sliders -padx 3m -pady 3m
pack .b_quit .b_save .b_load .b_go .b_stop .b_step \
	-in .buttons -side left -ipadx 2m -ipady 2m -padx 2m -pady 2m
pack .beta .eta .settle\
	-in .sliders -side left -ipadx 2m -ipady 2m -padx 2m -pady 2m

update

#
# do simulation
#
set run_p 0
while {1} {
    if {$run_p} {
	stepNet 100
    }
    update
}
