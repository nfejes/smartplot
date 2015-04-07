import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Default preferences
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=11)
plt.rc('legend', fontsize=11)


# Default plot colors, as of MATLAB 2014b
defcolors = [
	[0.000, 0.447, 0.741],
	[0.850, 0.325, 0.098],
	[0.929, 0.694, 0.125],
	[0.494, 0.184, 0.556],
	[0.466, 0.674, 0.188],
	[0.301, 0.745, 0.933],
	[0.635, 0.078, 0.184],
]


# Plot hold
__hold_sub = None


# Next color
__linecounter = 0
def __nextcolor():
	global __linecounter
	color = defcolors[__linecounter]
	__linecounter = (__linecounter + 1) % len(defcolors)
	return color


def texengform(val,n):
	"""
	texengform(val,n):
	Format a number [val] to TeX style with [n] decimals,
	e.g. texengform(1.43254e4,3) -> "$1.432\\times10^{4}$"
	"""
	v,e = (('%.'+str(n)+'e') % val).split('e')
	e = int(e,10)
	if e == 0: return '$%s$' % v
	else:      return '$%s\\times10^{%d}$' % (v,e)


def axform(sub,form,n):
	if n == 0:   get,set = sub.get_xticks,sub.set_xticklabels
	elif n == 1: get,set = sub.get_yticks,sub.set_yticklabels
	elif n == 2: get,set = sub.get_zticks,sub.set_zticklabels
	else: return
	if isinstance(form, str):
		if form.find('eng:') == 0:
			n = int(form[4:])
			set([texengform(i,n) for i in get()])
		else:
			set([(form % i) for i in get()])
	else:
		set([form(i) for i in get()])


# 2D/3D line plot
def smartplot(x,props={},insub=None):
	# Process multiple dicts
	if isinstance(props,(list,tuple)):
		pmerge = {}
		for p in props:
			pmerge.update(p)
		props = pmerge

	# getprop lambda
	getprop = lambda v,d=None: props[v] if v in props else d

	# Format x for most special cases
	if not hasattr(x,'__len__'): x = []
	if len(x) == 1: x = (range(len(x[0])),x[0])
	if len(x) > 3:  x = (range(len(x)),x)

	# Default range
	if len(x) > 0:
		arange = getprop('range',np.array([np.min(x,1),np.max(x,1)]).T)
	else:
		arange = []

	# Set colors
	color  = getprop('color',-1)
	mcolor = getprop('mcolor')
	if isinstance(color,int):
		global __linecounter
		if color == -1:
			color = __nextcolor()
		else:
			color = defcolors[color % len(defcolors)]
	if mcolor is None:
		mcolor = color

	# Modulate data
	if len(x) > 0:
		mod = getprop('mod')
		if mod:
			# TODO
			if getprop('linewidth'):
				pass
			m = mod[1] - mod[0]
			d = mod[0]
			x = np.array(x)
			x[0] = np.mod(x[0]-d,m) + d

	# Add margin
	# TODO fix
	#if getprop('margin'):
	#	print('a',arange)
	#	m = getprop('margin')
	#	axmin = np.min(arange,1).reshape([-1])
	#	axmax = np.max(arange,1).reshape([-1])
	#	axlen = abs(axmax-axmin)
	#	arange = np.array([axmin - axlen*m,axmax + axlen*m])
	#	print('b',arange)

	# Subplot
	if insub:
		sub = insub
	else:
		global __hold_sub
		# If hold, use prev sub
		if getprop('hold') and __hold_sub:
			sub = __hold_sub
			if getprop('hold') == 'clear':
				sub.clear()

		# If previous hold, use that
		elif __hold_sub:
			sub = __hold_sub
			__hold_sub = None

		# Else create subplot
		else:
			fig = plt.figure(figsize=getprop('figsize', (7.5,4)),dpi=getprop('figdpi',300))
			if len(x) == 3:
				sub = fig.add_subplot(111,projection='3d')
				plt.draw() # needed to create zticks
			else:
				sub = fig.add_subplot(111)

		# Store hold sub
		if getprop('hold') and not __hold_sub:
			__hold_sub = sub


	# Plot arguments
	kwargs = {
		'markersize' : getprop('markersize', 1),
		'linewidth'  : getprop('linewidth', 0.2),
		'linestyle'  : getprop('linestyle', '-'),
		'marker'     : getprop('marker', '.'),
	}
	if not color is None:  kwargs['color'] = color
	if not mcolor is None: kwargs['mfc'] = kwargs['mec'] = mcolor
	if getprop('label'): kwargs['label'] = getprop('label')

	# Expand range
	if len(x) > 0 and getprop('expand'):
		# TODO: find a way to count subplots
		if not all(np.array(sub.get_xlim()) == (0,1)):
			c = [sub.get_xlim(), sub.get_ylim()]
			if len(x) == 3: c += [sub.get_zlim()]
			r = np.concatenate([c,arange],1)
			arange = np.array([np.min(r,1),np.max(r,1)]).T

	# Plot
	if len(x):
		if len(x) == 3:
			print(sub)
			phandle = sub.plot(xs=x[0],ys=x[1],zs=x[2],**kwargs)
		else:
			phandle = sub.plot(x[0],x[1],**kwargs)
	else:
		phandle = None

	# Range
	if len(arange) > 0:
		sub.set_xlim(arange[0])
		sub.set_ylim(arange[1])
		if len(arange) == 3:
			sub.set_zlim(arange[2])

	# Tick handling
	def tickspace(x,mul):
		a = np.ceil(np.min(x) / mul)
		b = np.floor(np.max(x) / mul)
		return np.arange(a,b+1) * mul
	
	if len(arange) > 0:
		if 'xtick' in props: sub.set_xticks(tickspace(arange[0],props['xtick']))
		if 'ytick' in props: sub.set_yticks(tickspace(arange[1],props['ytick']))
		if len(arange) == 3 and 'ztick' in props:
			sub.set_zticks(tickspace(arange[2],props['ztick']))

	if getprop('title'): sub.set_title(getprop('title'))

	# Set labels
	labels = getprop('labels')
	if labels:
		if labels[0]: sub.set_xlabel(labels[0])
		if labels[1]: sub.set_ylabel(labels[1])
		if len(labels) == 3 and labels[2]:
			sub.set_zlabel(labels[2])

	if getprop('axform'):
		for n,form in enumerate(getprop('axform')):
			if form:
				axform(sub,form,n)

	if getprop('smallticks'):
		ticklabels = (sub.get_xticklabels() + sub.get_yticklabels())
		if len(x) == 3: ticklabels += sub.get_zticklabels()
		for item in ticklabels:
			item.set_fontsize(8)

	if getprop('prerender'):
		props['prerender'](sub)

	plt.tight_layout()

	if getprop('pdf'):
		plt.savefig(getprop('pdf'), format='pdf', dpi=1000)

	if getprop('png'):
		plt.savefig(getprop('png'), format='png', dpi=getprop('pngdpi',400))

	if getprop('pause'):
		plt.ion()
		plt.show()
		plt.pause(getprop('pause'))
	elif (insub is None and not getprop('hold')) or getprop('show',False):
		plt.show()

	return phandle

